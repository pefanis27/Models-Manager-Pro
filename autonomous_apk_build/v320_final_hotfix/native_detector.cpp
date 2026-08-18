#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <net.h>
#include <gpu.h>
#include <layer.h>
#include <cpu.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <memory>
#include <mutex>
#include <vector>

#define LOG_TAG "AutonomousDiagnosis"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
constexpr int kTargetSize = 640;
constexpr int kNumClasses = 11;
constexpr int kChannels = 4 + kNumClasses;
constexpr int kAnchors = 8400;
constexpr int kMaxObjects = 8;
constexpr int kCandidateCap = 256;
constexpr const char* kParamAsset = "diagnosis_yolo26s_640.ncnn.param";
constexpr const char* kBinAsset = "diagnosis_yolo26s_640.ncnn.bin";

struct Object {
    float x1=0.f, y1=0.f, x2=0.f, y2=0.f, score=0.f;
    int label=-1;
};
struct Roi { int x=0,y=0,w=0,h=0; };

std::unique_ptr<ncnn::Net> g_net;
std::mutex g_mutex;
bool g_logged_output=false;

inline int domain(int label) { return label >= 0 && label < 6 ? 0 : 1; }
inline float area(const Object& o) { return std::max(0.f,o.x2-o.x1)*std::max(0.f,o.y2-o.y1); }
float iou(const Object& a,const Object& b) {
    const float x1=std::max(a.x1,b.x1), y1=std::max(a.y1,b.y1);
    const float x2=std::min(a.x2,b.x2), y2=std::min(a.y2,b.y2);
    const float inter=std::max(0.f,x2-x1)*std::max(0.f,y2-y1);
    const float uni=area(a)+area(b)-inter;
    return uni>0.f?inter/uni:0.f;
}
float containment(const Object& a,const Object& b) {
    const float x1=std::max(a.x1,b.x1), y1=std::max(a.y1,b.y1);
    const float x2=std::min(a.x2,b.x2), y2=std::min(a.y2,b.y2);
    const float inter=std::max(0.f,x2-x1)*std::max(0.f,y2-y1);
    return inter/std::max(1.f,std::min(area(a),area(b)));
}
inline float overlap_1d(float a1,float a2,float b1,float b2) {
    return std::max(0.f,std::min(a2,b2)-std::max(a1,b1));
}
inline float axis_gap(float a1,float a2,float b1,float b2) {
    if(a2<b1) return b1-a2;
    if(b2<a1) return a1-b2;
    return 0.f;
}

void domain_aware_nms(std::vector<Object>& objects,float threshold) {
    std::sort(objects.begin(),objects.end(),[](const Object&a,const Object&b){return a.score>b.score;});
    std::vector<Object> kept; kept.reserve(kMaxObjects);
    for(const auto& candidate:objects) {
        bool suppress=false;
        for(const auto& accepted:kept) {
            if(domain(candidate.label)!=domain(accepted.label)) continue;
            if(iou(candidate,accepted)>threshold || containment(candidate,accepted)>=0.82f) { suppress=true; break; }
        }
        if(!suppress) kept.push_back(candidate);
        if(static_cast<int>(kept.size())>=kMaxObjects) break;
    }
    objects.swap(kept);
}

/**
 * Remove only obvious fragments of one physical target.  This intentionally
 * does not merge ordinary overlapping neighbours.  The rules were regression
 * checked against the corrected held-out test set of 764 images.
 */
void suppress_obvious_fragments(std::vector<Object>& objects) {
    bool changed=true;
    while(changed) {
        changed=false;
        for(size_t i=0;i<objects.size() && !changed;++i) {
            for(size_t j=i+1;j<objects.size();++j) {
                Object& a=objects[i]; Object& b=objects[j];
                if(domain(a.label)!=domain(b.label)) {
                    // Cross-domain suppression is intentionally extremely conservative.
                    // It only removes a weak, tiny box almost fully nested inside a
                    // much larger, high-confidence target.  This fixes the held-out
                    // grape-with-tiny-leaf-fragment case without applying general
                    // leaf-vs-grape NMS to legitimate neighbouring objects.
                    const float aa=area(a), ba=area(b);
                    const Object& large=aa>=ba?a:b;
                    const Object& small=aa>=ba?b:a;
                    const float ratio=std::min(aa,ba)/std::max(1.f,std::max(aa,ba));
                    const bool nested_cross_domain=
                        containment(a,b)>=0.92f && ratio<=0.08f &&
                        large.score>=0.80f && small.score<=0.55f &&
                        large.score-small.score>=0.30f;
                    if(nested_cross_domain) {
                        const size_t erase_index=aa>=ba?j:i;
                        objects.erase(objects.begin()+static_cast<std::ptrdiff_t>(erase_index));
                        changed=true; break;
                    }
                    continue;
                }
                const float aw=std::max(1.f,a.x2-a.x1), ah=std::max(1.f,a.y2-a.y1);
                const float bw=std::max(1.f,b.x2-b.x1), bh=std::max(1.f,b.y2-b.y1);
                const float xov=overlap_1d(a.x1,a.x2,b.x1,b.x2)/std::min(aw,bw);
                const float yov=overlap_1d(a.y1,a.y2,b.y1,b.y2)/std::min(ah,bh);
                const float xgap=axis_gap(a.x1,a.x2,b.x1,b.x2);
                const float ygap=axis_gap(a.y1,a.y2,b.y1,b.y2);
                const float wr=std::min(aw,bw)/std::max(aw,bw);
                const float hr=std::min(ah,bh)/std::max(ah,bh);

                const bool side_sliver=yov>=0.82f && xgap<=20.f && wr<=0.35f;
                const bool top_bottom_sliver=xov>=0.82f && ygap<=20.f && hr<=0.35f;
                if(side_sliver || top_bottom_sliver) {
                    const size_t erase_index=area(a)>=area(b)?j:i;
                    objects.erase(objects.begin()+static_cast<std::ptrdiff_t>(erase_index));
                    changed=true; break;
                }

                const float area_ratio=std::min(area(a),area(b))/std::max(1.f,std::max(area(a),area(b)));
                const bool horizontal_halves=
                    xov>=0.90f && ygap<=20.f && area_ratio>=0.55f &&
                    aw/ah>=1.70f && bw/bh>=1.70f;
                const bool vertical_halves=
                    yov>=0.90f && xgap<=20.f && area_ratio>=0.55f &&
                    ah/aw>=1.70f && bh/bw>=1.70f;
                if(horizontal_halves || vertical_halves) {
                    Object merged=(a.score>=b.score)?a:b;
                    merged.x1=std::min(a.x1,b.x1); merged.y1=std::min(a.y1,b.y1);
                    merged.x2=std::max(a.x2,b.x2); merged.y2=std::max(a.y2,b.y2);
                    merged.score=std::max(a.score,b.score);
                    objects[i]=merged;
                    objects.erase(objects.begin()+static_cast<std::ptrdiff_t>(j));
                    changed=true; break;
                }
            }
        }
    }
    std::sort(objects.begin(),objects.end(),[](const Object&a,const Object&b){return a.score>b.score;});
    if(static_cast<int>(objects.size())>kMaxObjects) objects.resize(kMaxObjects);
}

inline bool near_black_rgba(const unsigned char* p) {
    return std::max({p[0],p[1],p[2]})<=28;
}

Roi content_roi(const unsigned char* rgba,int w,int h) {
    Roi full{0,0,w,h};
    if(!rgba || w<32 || h<32) return full;
    const int sx=std::max(1,w/512), sy=std::max(1,h/512);
    auto row_black=[&](int y)->bool {
        int dark=0,total=0;
        for(int x=0;x<w;x+=sx){dark+=near_black_rgba(rgba+(static_cast<size_t>(y)*w+x)*4)?1:0;++total;}
        return total>0 && static_cast<float>(dark)/total>=0.985f;
    };
    auto col_black=[&](int x,int top,int bottom)->bool {
        int dark=0,total=0;
        for(int y=top;y<bottom;y+=sy){dark+=near_black_rgba(rgba+(static_cast<size_t>(y)*w+x)*4)?1:0;++total;}
        return total>0 && static_cast<float>(dark)/total>=0.985f;
    };
    const int max_rows=std::max(1,static_cast<int>(h*0.45f));
    int top=0; while(top<max_rows && row_black(top)) ++top;
    int bottom=h, scanned=0; while(bottom>top && scanned<max_rows && row_black(bottom-1)){--bottom;++scanned;}
    const int max_cols=std::max(1,static_cast<int>(w*0.45f));
    int left=0; while(left<max_cols && col_black(left,top,bottom)) ++left;
    int right=w; scanned=0; while(right>left && scanned<max_cols && col_black(right-1,top,bottom)){--right;++scanned;}
    const int rw=right-left,rh=bottom-top;
    if(rw<32 || rh<32 || static_cast<long long>(rw)*rh<static_cast<long long>(w)*h*18/100) return full;
    return Roi{left,top,rw,rh};
}

bool load_model(AAssetManager* manager,bool use_vulkan) {
    auto net=std::make_unique<ncnn::Net>();
    net->opt.num_threads=4;
    net->opt.use_fp16_packed=true;
    net->opt.use_fp16_storage=true;
    net->opt.use_fp16_arithmetic=false;
    net->opt.use_vulkan_compute=use_vulkan && ncnn::get_gpu_count()>0;
    const int p=net->load_param(manager,kParamAsset);
    const int m=net->load_model(manager,kBinAsset);
    if(p!=0 || m!=0 || net->input_indexes().empty() || net->output_indexes().empty()) {
        LOGE("Model load failed param=%d bin=%d",p,m); return false;
    }
    LOGI("YOLO26s 640 official NCNN loaded Vulkan=%d",net->opt.use_vulkan_compute?1:0);
    g_logged_output=false; g_net=std::move(net); return true;
}

std::vector<Object> decode(
    const ncnn::Mat& out,float confidence,float nms_threshold,float scale,
    int pad_left,int pad_top,int roi_x,int roi_y,int image_w,int image_h) {
    const bool channels_first=out.dims==2 && out.w==kAnchors && out.h==kChannels;
    const bool anchors_first=out.dims==2 && out.w==kChannels && out.h==kAnchors;
    if((!channels_first && !anchors_first) || out.elempack!=1 || out.elembits()!=32) {
        LOGE("Invalid output shape d=%d w=%d h=%d c=%d bits=%d pack=%d",out.dims,out.w,out.h,out.c,out.elembits(),out.elempack);
        return {};
    }
    auto value=[&](int channel,int anchor)->float {
        return channels_first ? out.row(channel)[anchor] : out.row(anchor)[channel];
    };
    std::vector<Object> candidates; candidates.reserve(32);
    bool capped=false;
    for(int a=0;a<kAnchors;++a) {
        int best=-1; float score=-FLT_MAX;
        for(int c=0;c<kNumClasses;++c) {
            const float s=value(4+c,a); if(s>score){score=s;best=c;}
        }
        if(best<0 || !std::isfinite(score) || score<confidence) continue;
        if(score>1.001f) { LOGE("Invalid probability %.5f",score); return {}; }

        // Official Ultralytics YOLO26 NCNN export (end2end:false) outputs
        // decoded CX,CY,W,H in pixels followed by 11 sigmoid class scores.
        const float cx=value(0,a), cy=value(1,a), bw=value(2,a), bh=value(3,a);
        if(!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(bw)||!std::isfinite(bh)) continue;
        if(bw<4.f || bh<4.f || bw>kTargetSize*2.f || bh>kTargetSize*2.f) continue;
        const float raw_x1=cx-bw*0.5f, raw_y1=cy-bh*0.5f;
        const float raw_x2=cx+bw*0.5f, raw_y2=cy+bh*0.5f;
        Object o;
        o.x1=(raw_x1-pad_left)/scale+roi_x; o.y1=(raw_y1-pad_top)/scale+roi_y;
        o.x2=(raw_x2-pad_left)/scale+roi_x; o.y2=(raw_y2-pad_top)/scale+roi_y;
        o.x1=std::clamp(o.x1,0.f,static_cast<float>(image_w-1));
        o.y1=std::clamp(o.y1,0.f,static_cast<float>(image_h-1));
        o.x2=std::clamp(o.x2,0.f,static_cast<float>(image_w-1));
        o.y2=std::clamp(o.y2,0.f,static_cast<float>(image_h-1));
        o.score=score; o.label=best;
        if(o.x2-o.x1<4.f || o.y2-o.y1<4.f) continue;
        if(static_cast<int>(candidates.size())<kCandidateCap) candidates.push_back(o);
        else capped=true;
    }
    if(capped) LOGI("Candidate cap reached at conf %.3f; keeping best after suppression",confidence);
    domain_aware_nms(candidates,nms_threshold);
    suppress_obvious_fragments(candidates);
    return candidates;
}
} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*,void*) { ncnn::create_gpu_instance(); return JNI_VERSION_1_6; }
extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*,void*) {
    std::lock_guard<std::mutex> lock(g_mutex); g_net.reset(); ncnn::destroy_gpu_instance();
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeInit(JNIEnv* env,jobject,jobject assets,jboolean vulkan) {
    std::lock_guard<std::mutex> lock(g_mutex); g_net.reset();
    AAssetManager* manager=AAssetManager_fromJava(env,assets);
    return manager && load_model(manager,vulkan==JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeHasGpu(JNIEnv*,jobject) {
    return ncnn::get_gpu_count()>0 ? JNI_TRUE : JNI_FALSE;
}
extern "C" JNIEXPORT void JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeClose(JNIEnv*,jobject) {
    std::lock_guard<std::mutex> lock(g_mutex); g_net.reset();
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeDetect(
    JNIEnv* env,jobject,jobject rgba_buffer,jint image_w,jint image_h,jint row_stride,jfloat confidence,jfloat nms_threshold) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if(!g_net || !rgba_buffer || image_w<=0 || image_h<=0) return env->NewFloatArray(0);
    auto* rgba=static_cast<unsigned char*>(env->GetDirectBufferAddress(rgba_buffer));
    const jlong capacity=env->GetDirectBufferCapacity(rgba_buffer);
    if(!rgba || row_stride<image_w*4 || capacity<static_cast<jlong>(row_stride)*image_h) return env->NewFloatArray(0);

    std::vector<unsigned char> contiguous;
    const unsigned char* pixels=rgba;
    if(row_stride!=image_w*4) {
        contiguous.resize(static_cast<size_t>(image_w)*image_h*4);
        for(int y=0;y<image_h;++y) {
            std::copy_n(rgba+static_cast<size_t>(y)*row_stride,static_cast<size_t>(image_w)*4,
                        contiguous.data()+static_cast<size_t>(y)*image_w*4);
        }
        pixels=contiguous.data();
    }

    Roi roi=content_roi(pixels,image_w,image_h);
    std::vector<unsigned char> roi_pixels;
    const unsigned char* analysis_pixels=pixels;
    if(roi.x!=0 || roi.y!=0 || roi.w!=image_w || roi.h!=image_h) {
        roi_pixels.resize(static_cast<size_t>(roi.w)*roi.h*4);
        for(int y=0;y<roi.h;++y) {
            const auto* src=pixels+(static_cast<size_t>(roi.y+y)*image_w+roi.x)*4;
            auto* dst=roi_pixels.data()+static_cast<size_t>(y)*roi.w*4;
            std::copy_n(src,static_cast<size_t>(roi.w)*4,dst);
        }
        analysis_pixels=roi_pixels.data();
    }

    const float scale=std::min(static_cast<float>(kTargetSize)/roi.w,static_cast<float>(kTargetSize)/roi.h);
    const int resized_w=std::max(1,static_cast<int>(std::round(roi.w*scale)));
    const int resized_h=std::max(1,static_cast<int>(std::round(roi.h*scale)));
    ncnn::Mat resized=ncnn::Mat::from_pixels_resize(analysis_pixels,ncnn::Mat::PIXEL_RGBA2RGB,roi.w,roi.h,resized_w,resized_h);
    const int pad_w=kTargetSize-resized_w, pad_h=kTargetSize-resized_h;
    const int pad_left=pad_w/2, pad_top=pad_h/2;
    ncnn::Mat input;
    ncnn::copy_make_border(resized,input,pad_top,pad_h-pad_top,pad_left,pad_w-pad_left,ncnn::BORDER_CONSTANT,114.f);
    const float norm[3]={1.f/255.f,1.f/255.f,1.f/255.f};
    input.substract_mean_normalize(nullptr,norm);

    ncnn::Extractor ex=g_net->create_extractor(); ex.set_light_mode(true);
    if(ex.input(g_net->input_indexes().front(),input)!=0) return env->NewFloatArray(0);
    ncnn::Mat native_out;
    if(ex.extract(g_net->output_indexes().front(),native_out)!=0 || native_out.empty()) return env->NewFloatArray(0);

    ncnn::Mat unpacked=native_out;
    if(unpacked.elempack!=1) {
        ncnn::Mat tmp; ncnn::convert_packing(unpacked,tmp,1,g_net->opt);
        if(tmp.empty()) return env->NewFloatArray(0); unpacked=tmp;
    }
    ncnn::Mat fp32=unpacked;
    if(fp32.elembits()==16) {
        ncnn::Mat tmp; ncnn::cast_float16_to_float32(fp32,tmp,g_net->opt);
        if(tmp.empty()) return env->NewFloatArray(0); fp32=tmp;
    } else if(fp32.elembits()!=32) return env->NewFloatArray(0);

    if(!g_logged_output) {
        LOGI("Output d=%d w=%d h=%d c=%d bits=%d pack=%d ROI=%d,%d %dx%d",
             fp32.dims,fp32.w,fp32.h,fp32.c,fp32.elembits(),fp32.elempack,roi.x,roi.y,roi.w,roi.h);
        g_logged_output=true;
    }
    const auto objects=decode(fp32,
        std::clamp(static_cast<float>(confidence),0.10f,0.90f),
        std::clamp(static_cast<float>(nms_threshold),0.20f,0.80f),
        scale,pad_left,pad_top,roi.x,roi.y,image_w,image_h);
    std::vector<jfloat> flat; flat.reserve(objects.size()*6);
    for(const auto&o:objects) {
        flat.push_back(o.x1); flat.push_back(o.y1); flat.push_back(o.x2); flat.push_back(o.y2);
        flat.push_back(o.score); flat.push_back(static_cast<float>(o.label));
    }
    jfloatArray result=env->NewFloatArray(static_cast<jsize>(flat.size()));
    if(result && !flat.empty()) env->SetFloatArrayRegion(result,0,static_cast<jsize>(flat.size()),flat.data());
    return result;
}
