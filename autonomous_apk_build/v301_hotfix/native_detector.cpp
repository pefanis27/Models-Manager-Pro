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
constexpr int kTargetSize = 512;
constexpr int kNumClasses = 11;
constexpr int kChannels = 4 + kNumClasses;

struct Object {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    float score = 0.f;
    int label = -1;
};

std::unique_ptr<ncnn::Net> g_net;
std::mutex g_mutex;
bool g_using_vulkan = false;

float area(const Object& o) {
    return std::max(0.f, o.x2 - o.x1) * std::max(0.f, o.y2 - o.y1);
}

float iou(const Object& a, const Object& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float uni = area(a) + area(b) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

void class_agnostic_nms(std::vector<Object>& objects, float threshold) {
    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.score > b.score;
    });
    std::vector<Object> kept;
    kept.reserve(objects.size());
    for (const auto& candidate : objects) {
        bool suppress = false;
        for (const auto& accepted : kept) {
            if (iou(candidate, accepted) > threshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) kept.push_back(candidate);
        if (kept.size() >= 8) break;
    }
    objects.swap(kept);
}

bool decoded_layout(const ncnn::Mat& out, int& anchors, int& layout) {
    if (out.dims >= 2 && out.h == kChannels) {
        anchors = out.w;
        layout = 0;
        return anchors > 0;
    }
    if (out.dims >= 2 && out.w == kChannels) {
        anchors = out.h * std::max(1, out.c);
        layout = 1;
        return anchors > 0;
    }
    if (out.dims == 3 && out.c == kChannels) {
        anchors = out.w * out.h;
        layout = 2;
        return anchors > 0;
    }
    return false;
}

float pred_value(const ncnn::Mat& out, int layout, int channel, int anchor) {
    if (layout == 0) return out.row(channel)[anchor];
    if (layout == 1) {
        const int row = anchor % out.h;
        const int channel_block = anchor / out.h;
        if (out.c <= 1) return out.row(row)[channel];
        return out.channel(channel_block).row(row)[channel];
    }
    const int y = anchor / out.w;
    const int x = anchor % out.w;
    return out.channel(channel).row(y)[x];
}

std::vector<Object> parse_decoded_output(
    const ncnn::Mat& out,
    float confidence,
    float nms_threshold,
    float scale,
    int pad_left,
    int pad_top,
    int image_w,
    int image_h) {

    int anchors = 0;
    int layout = -1;
    if (!decoded_layout(out, anchors, layout)) {
        LOGE("Unexpected NCNN output shape dims=%d w=%d h=%d c=%d", out.dims, out.w, out.h, out.c);
        return {};
    }

    if (anchors != 5376 || layout != 0) {
        LOGE("Unexpected decoded tensor layout anchors=%d layout=%d; expected [15,5376]", anchors, layout);
        return {};
    }

    std::vector<Object> objects;
    objects.reserve(64);
    int candidates_above_threshold = 0;

    for (int a = 0; a < anchors; ++a) {
        int best_class = -1;
        float best_score = -FLT_MAX;
        for (int c = 0; c < kNumClasses; ++c) {
            const float score = pred_value(out, layout, 4 + c, a);
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_class < 0 || !std::isfinite(best_score) || best_score < confidence) continue;
        if (best_score > 1.001f) {
            LOGE("Invalid class probability %.6f at anchor=%d", best_score, a);
            return {};
        }
        if (++candidates_above_threshold > 64) {
            LOGE("Detection flood guard: >64 candidates above confidence %.3f; dropping frame", confidence);
            return {};
        }

        const float cx = pred_value(out, layout, 0, a);
        const float cy = pred_value(out, layout, 1, a);
        const float w = pred_value(out, layout, 2, a);
        const float h = pred_value(out, layout, 3, a);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) ||
            w <= 0.f || h <= 0.f || w > kTargetSize * 3.f || h > kTargetSize * 3.f) {
            continue;
        }

        Object obj;
        obj.x1 = (cx - w * 0.5f - pad_left) / scale;
        obj.y1 = (cy - h * 0.5f - pad_top) / scale;
        obj.x2 = (cx + w * 0.5f - pad_left) / scale;
        obj.y2 = (cy + h * 0.5f - pad_top) / scale;
        obj.x1 = std::clamp(obj.x1, 0.f, static_cast<float>(image_w - 1));
        obj.y1 = std::clamp(obj.y1, 0.f, static_cast<float>(image_h - 1));
        obj.x2 = std::clamp(obj.x2, 0.f, static_cast<float>(image_w - 1));
        obj.y2 = std::clamp(obj.y2, 0.f, static_cast<float>(image_h - 1));
        obj.score = best_score;
        obj.label = best_class;
        if (obj.x2 > obj.x1 && obj.y2 > obj.y1) objects.push_back(obj);
    }

    class_agnostic_nms(objects, nms_threshold);
    return objects;
}

bool load_model(AAssetManager* manager, bool use_vulkan) {
    auto net = std::make_unique<ncnn::Net>();
    const int big_cores = std::max(1, ncnn::get_big_cpu_count());
    net->opt.num_threads = std::min(4, big_cores);
    net->opt.use_fp16_packed = true;
    net->opt.use_fp16_storage = true;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_vulkan_compute = use_vulkan && ncnn::get_gpu_count() > 0;

    const int p = net->load_param(manager, "autonomous_diagnosis.ncnn.param");
    const int m = net->load_model(manager, "autonomous_diagnosis.ncnn.bin");
    if (p != 0 || m != 0) {
        LOGE("Model load failed param=%d model=%d", p, m);
        return false;
    }
    if (net->input_indexes().empty() || net->output_indexes().empty()) {
        LOGE("Model has no input/output blobs");
        return false;
    }
    g_using_vulkan = net->opt.use_vulkan_compute;
    g_net = std::move(net);
    LOGI("Model loaded. Vulkan=%d inputs=%zu outputs=%zu", g_using_vulkan ? 1 : 0,
         g_net->input_indexes().size(), g_net->output_indexes().size());
    return true;
}
} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    ncnn::create_gpu_instance();
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    std::lock_guard<std::mutex> guard(g_mutex);
    g_net.reset();
    ncnn::destroy_gpu_instance();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeInit(
    JNIEnv* env, jobject, jobject asset_manager, jboolean use_vulkan) {
    std::lock_guard<std::mutex> guard(g_mutex);
    g_net.reset();
    AAssetManager* manager = AAssetManager_fromJava(env, asset_manager);
    if (!manager) return JNI_FALSE;
    return load_model(manager, use_vulkan == JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeHasGpu(JNIEnv*, jobject) {
    return ncnn::get_gpu_count() > 0 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeDetect(
    JNIEnv* env,
    jobject,
    jobject rgba_buffer,
    jint image_w,
    jint image_h,
    jint row_stride,
    jfloat confidence,
    jfloat nms_threshold) {

    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_net || !rgba_buffer || image_w <= 0 || image_h <= 0) return env->NewFloatArray(0);

    auto* rgba = static_cast<unsigned char*>(env->GetDirectBufferAddress(rgba_buffer));
    const jlong capacity = env->GetDirectBufferCapacity(rgba_buffer);
    if (!rgba || capacity < static_cast<jlong>(row_stride) * image_h || row_stride < image_w * 4) {
        LOGE("Invalid RGBA buffer capacity=%lld stride=%d image=%dx%d", static_cast<long long>(capacity), row_stride, image_w, image_h);
        return env->NewFloatArray(0);
    }

    std::vector<unsigned char> contiguous;
    const unsigned char* pixels = rgba;
    if (row_stride != image_w * 4) {
        contiguous.resize(static_cast<size_t>(image_w) * image_h * 4);
        for (int y = 0; y < image_h; ++y) {
            std::copy_n(rgba + static_cast<size_t>(y) * row_stride,
                        static_cast<size_t>(image_w) * 4,
                        contiguous.data() + static_cast<size_t>(y) * image_w * 4);
        }
        pixels = contiguous.data();
    }

    const float scale = std::min(
        static_cast<float>(kTargetSize) / image_w,
        static_cast<float>(kTargetSize) / image_h);
    const int resized_w = std::max(1, static_cast<int>(std::round(image_w * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(image_h * scale)));

    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        pixels,
        ncnn::Mat::PIXEL_RGBA2RGB,
        image_w,
        image_h,
        resized_w,
        resized_h);

    const int pad_w = kTargetSize - resized_w;
    const int pad_h = kTargetSize - resized_h;
    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    ncnn::Mat input;
    ncnn::copy_make_border(resized, input, pad_top, pad_bottom, pad_left, pad_right, ncnn::BORDER_CONSTANT, 114.f);

    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor ex = g_net->create_extractor();
    const int input_index = g_net->input_indexes().front();
    const int output_index = g_net->output_indexes().front();
    if (ex.input(input_index, input) != 0) {
        LOGE("NCNN input failed");
        return env->NewFloatArray(0);
    }

    ncnn::Mat output_native;
    if (ex.extract(output_index, output_native) != 0 || output_native.empty()) {
        LOGE("NCNN output extraction failed");
        return env->NewFloatArray(0);
    }

    ncnn::Mat output_unpacked = output_native;
    if (output_unpacked.elempack != 1) {
        ncnn::Mat tmp;
        ncnn::convert_packing(output_unpacked, tmp, 1, g_net->opt);
        if (tmp.empty()) {
            LOGE("NCNN output unpack failed pack=%d", output_native.elempack);
            return env->NewFloatArray(0);
        }
        output_unpacked = tmp;
    }

    ncnn::Mat output_fp32 = output_unpacked;
    if (output_fp32.elembits() == 16) {
        ncnn::Mat tmp;
        ncnn::cast_float16_to_float32(output_fp32, tmp, g_net->opt);
        if (tmp.empty()) {
            LOGE("NCNN FP16->FP32 output cast failed");
            return env->NewFloatArray(0);
        }
        output_fp32 = tmp;
    } else if (output_fp32.elembits() != 32) {
        LOGE("Unexpected output elembits=%d", output_fp32.elembits());
        return env->NewFloatArray(0);
    }

    if (output_fp32.dims != 2 || output_fp32.w != 5376 || output_fp32.h != kChannels || output_fp32.elempack != 1) {
        LOGE("Unexpected normalized output dims=%d w=%d h=%d c=%d bits=%d pack=%d",
             output_fp32.dims, output_fp32.w, output_fp32.h, output_fp32.c,
             output_fp32.elembits(), output_fp32.elempack);
        return env->NewFloatArray(0);
    }

    auto objects = parse_decoded_output(
        output_fp32,
        std::clamp(static_cast<float>(confidence), 0.05f, 0.95f),
        std::clamp(static_cast<float>(nms_threshold), 0.10f, 0.90f),
        scale, pad_left, pad_top, image_w, image_h);

    std::vector<jfloat> flattened;
    flattened.reserve(objects.size() * 6);
    for (const auto& o : objects) {
        flattened.push_back(o.x1);
        flattened.push_back(o.y1);
        flattened.push_back(o.x2);
        flattened.push_back(o.y2);
        flattened.push_back(o.score);
        flattened.push_back(static_cast<float>(o.label));
    }

    jfloatArray result = env->NewFloatArray(static_cast<jsize>(flattened.size()));
    if (result && !flattened.empty()) {
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(flattened.size()), flattened.data());
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_autonomousdiagnosis_app_NativeDetector_nativeClose(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> guard(g_mutex);
    g_net.reset();
    g_using_vulkan = false;
}
