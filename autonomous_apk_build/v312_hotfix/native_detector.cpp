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
constexpr int kFloodGuard = 128;
constexpr const char* kParamAsset = "unified_grapevine_yolo26s_640.ncnn.param";
constexpr const char* kBinAsset = "unified_grapevine_yolo26s_640.ncnn.bin";

struct Object {
    float x1=0.f, y1=0.f, x2=0.f, y2=0.f, score=0.f;
    int label=-1;
};

std::unique_ptr<ncnn::Net> g_net;
std::mutex g_mutex;

inline int domain(int label) { return label >= 0 && label < 6 ? 0 : 1; }
inline float area(const Object& o) {
    return std::max(0.f,o.x2-o.x1)*std::max(0.f,o.y2-o.y1);
}
float iou(const Object& a,const Object& b) {
    const float x1=std::max(a.x1,b.x1), y1=std::max(a.y1,b.y1);
    const float x2=std::min(a.x2,b.x2), y2=std::min(a.y2,b.y2);
    const float inter=std::max(0.f,x2-x1)*std::max(0.f,y2-y1);
    const float uni=area(a)+area(b)-inter;
    return uni>0.f?inter/uni:0.f;
}

void domain_aware_nms(std::vector<Object>& objects,float threshold) {
    std::sort(objects.begin(),objects.end(),[](const Object&a,const Object&b){return a.score>b.score;});
    std::vector<Object> kept;
    kept.reserve(kMaxObjects);
    for(const auto& candidate:objects) {
        bool suppress=false;
        for(const auto& accepted:kept) {
            if(domain(candidate.label)==domain(accepted.label) && iou(candidate,accepted)>threshold) {
                suppress=true; break;
            }
        }
        if(!suppress) kept.push_back(candidate);
        if(static_cast<int>(kept.size())>=kMaxObjects) break;
    }
    objects.swap(kept);
}

inline float overlap_1d(float a1,float a2,float b1,float b2) {
    return std::max(0.f,std::min(a2,b2)-std::max(a1,b1));
}
inline float axis_gap(float a1,float a2,float b1,float b2) {
    if(a2<b1) return b1-a2;
    if(b2<a1) return a1-b2;
    return 0.f;
}

/**
 * Merge the rare one-to-one-head fragmentation case where one physical target is
 * emitted as two touching strips of the SAME class. This is deliberately stricter
 * than NMS so two normal neighbouring leaves/clusters remain independent.
 */
bool aligned_same_class_fragments(const Object& a,const Object& b) {
    if(a.label!=b.label) return false;
    const float aw=std::max(1.f,a.x2-a.x1), ah=std::max(1.f,a.y2-a.y1);
    const float bw=std::max(1.f,b.x2-b.x1), bh=std::max(1.f,b.y2-b.y1);
    const float xov=overlap_1d(a.x1,a.x2,b.x1,b.x2)/std::min(aw,bw);
    const float yov=overlap_1d(a.y1,a.y2,b.y1,b.y2)/std::min(ah,bh);
    const float xgap=axis_gap(a.x1,a.x2,b.x1,b.x2);
    const float ygap=axis_gap(a.y1,a.y2,b.y1,b.y2);

    const float width_ratio=std::min(aw,bw)/std::max(aw,bw);
    const bool side_sliver=
        yov>=0.82f &&
        xgap<=std::max(18.f,0.065f*std::min(ah,bh)) &&
        width_ratio<=0.40f;

    const bool horizontal_strips=
        xov>=0.82f &&
        ygap<=std::max(18.f,0.065f*std::min(aw,bw)) &&
        (aw/ah)>=1.35f &&
        (bw/bh)>=1.35f;

    return side_sliver || horizontal_strips;
}

void merge_aligned_fragments(std::vector<Object>& objects) {
    bool changed=true;
    while(changed) {
        changed=false;
        for(size_t i=0;i<objects.size() && !changed;++i) {
            for(size_t j=i+1;j<objects.size();++j) {
                if(!aligned_same_class_fragments(objects[i],objects[j])) continue;
                Object merged=objects[i];
                merged.x1=std::min(objects[i].x1,objects[j].x1);
                merged.y1=std::min(objects[i].y1,objects[j].y1);
                merged.x2=std::max(objects[i].x2,objects[j].x2);
                merged.y2=std::max(objects[i].y2,objects[j].y2);
                merged.score=std::max(objects[i].score,objects[j].score);
                objects[i]=merged;
                objects.erase(objects.begin()+static_cast<std::ptrdiff_t>(j));
                changed=true;
                break;
            }
        }
    }
    std::sort(objects.begin(),objects.end(),[](const Object&a,const Object&b){return a.score>b.score;});
}

bool load_model(AAssetManager* manager,bool use_vulkan) {
    auto net=std::make_unique<ncnn::Net>();
    net->opt.num_threads=std::min(4,std::max(1,ncnn::get_big_cpu_count()));
    net->opt.use_fp16_packed=true;
    net->opt.use_fp16_storage=true;
    net->opt.use_fp16_arithmetic=false;
    net->opt.use_vulkan_compute=use_vulkan && ncnn::get_gpu_count()>0;
    const int p=net->load_param(manager,kParamAsset);
    const int m=net->load_model(manager,kBinAsset);
    if(p!=0 || m!=0 || net->input_indexes().empty() || net->output_indexes().empty()) {
        LOGE("Model load failed param=%d bin=%d",p,m); return false;
    }
    LOGI("YOLO26s 640 model loaded Vulkan=%d",net->opt.use_vulkan_compute?1:0);
    g_net=std::move(net); return true;
}

std::vector<Object> decode(
    const ncnn::Mat& out,float confidence,float nms_threshold,float scale,
    int pad_left,int pad_top,int image_w,int image_h) {
    if(out.dims!=2 || out.w!=kAnchors || out.h!=kChannels || out.elempack!=1 || out.elembits()!=32) {
        LOGE("Invalid output shape d=%d w=%d h=%d c=%d bits=%d pack=%d",out.dims,out.w,out.h,out.c,out.elembits(),out.elempack);
        return {};
    }
    std::vector<Object> candidates;
    candidates.reserve(32);
    for(int a=0;a<kAnchors;++a) {
        int best=-1; float score=-FLT_MAX;
        for(int c=0;c<kNumClasses;++c) {
            const float s=out.row(4+c)[a];
            if(s>score){score=s;best=c;}
        }
        if(best<0 || !std::isfinite(score) || score<confidence) continue;
        if(score>1.001f) { LOGE("Invalid probability %.5f",score); return {}; }
        if(static_cast<int>(candidates.size())>=kFloodGuard) {
            LOGE("Flood guard: >=%d candidates at conf %.3f",kFloodGuard,confidence); return {};
        }
        const float raw_x1=out.row(0)[a], raw_y1=out.row(1)[a];
        const float raw_x2=out.row(2)[a], raw_y2=out.row(3)[a];
        if(!std::isfinite(raw_x1)||!std::isfinite(raw_y1)||!std::isfinite(raw_x2)||!std::isfinite(raw_y2)) continue;
        if(raw_x2-raw_x1<4.f || raw_y2-raw_y1<4.f) continue;
        Object o;
        o.x1=(raw_x1-pad_left)/scale; o.y1=(raw_y1-pad_top)/scale;
        o.x2=(raw_x2-pad_left)/scale; o.y2=(raw_y2-pad_top)/scale;
        o.x1=std::clamp(o.x1,0.f,static_cast<float>(image_w-1));
        o.y1=std::clamp(o.y1,0.f,static_cast<float>(image_h-1));
        o.x2=std::clamp(o.x2,0.f,static_cast<float>(image_w-1));
        o.y2=std::clamp(o.y2,0.f,static_cast<float>(image_h-1));
        o.score=score; o.label=best;
        if(o.x2-o.x1>=4.f && o.y2-o.y1>=4.f) candidates.push_back(o);
    }
    domain_aware_nms(candidates,nms_threshold);
    merge_aligned_fragments(candidates);
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

    const float scale=std::min(static_cast<float>(kTargetSize)/image_w,static_cast<float>(kTargetSize)/image_h);
    const int resized_w=std::max(1,static_cast<int>(std::round(image_w*scale)));
    const int resized_h=std::max(1,static_cast<int>(std::round(image_h*scale)));
    ncnn::Mat resized=ncnn::Mat::from_pixels_resize(pixels,ncnn::Mat::PIXEL_RGBA2RGB,image_w,image_h,resized_w,resized_h);
    const int pad_w=kTargetSize-resized_w, pad_h=kTargetSize-resized_h;
    const int pad_left=pad_w/2, pad_top=pad_h/2;
    ncnn::Mat input;
    ncnn::copy_make_border(resized,input,pad_top,pad_h-pad_top,pad_left,pad_w-pad_left,ncnn::BORDER_CONSTANT,114.f);
    const float norm[3]={1.f/255.f,1.f/255.f,1.f/255.f};
    input.substract_mean_normalize(nullptr,norm);

    ncnn::Extractor ex=g_net->create_extractor();
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

    const auto objects=decode(fp32,
        std::clamp(static_cast<float>(confidence),0.10f,0.90f),
        std::clamp(static_cast<float>(nms_threshold),0.20f,0.80f),
        scale,pad_left,pad_top,image_w,image_h);
    std::vector<jfloat> flat; flat.reserve(objects.size()*6);
    for(const auto&o:objects) {
        flat.push_back(o.x1); flat.push_back(o.y1); flat.push_back(o.x2); flat.push_back(o.y2);
        flat.push_back(o.score); flat.push_back(static_cast<float>(o.label));
    }
    jfloatArray result=env->NewFloatArray(static_cast<jsize>(flat.size()));
    if(result && !flat.empty()) env->SetFloatArrayRegion(result,0,static_cast<jsize>(flat.size()),flat.data());
    return result;
}
