/**
 * @file yolo_rknn.cpp
 * @brief YOLO11-OBB RKNN 推理实现
 *
 * 封装 RKNN Lite2 API:
 *   - load:    rknn_init → 查询输出 shape → 分配缓冲区
 *   - detect:  preprocess → rknn_run → 拼接 4 个 tensor → decode
 *   - release: rknn_destroy
 *
 * 输入: NHWC uint8 [0, 255] (RKNN config 中 mean=0, std=255 自动归一化)
 * 输出: P2/P3/P4/P5 四个 tensor，每个 [1, 31, anchors_i]
 *       拼接后 [1, 31, 34000] (mode C, 已解码)
 */

#include "yolo_rknn.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

#ifndef RKNN_STUB
#include "rknn_api.h"
#endif

namespace yolo_rknn {

// ======================== 原始输出 dump 工具 ========================

/// 扫描所有 anchor，返回 max_cls_score > thresh 的 top-N 个 anchor 索引
static std::vector<int> find_top_anchors(const float* buf, int attrs, int total_anchors,
                                          int nc, float thresh, int top_n) {
    struct AnchorScore { int idx; float score; };
    std::vector<AnchorScore> candidates;
    for (int i = 0; i < total_anchors; i++) {
        float max_s = 0;
        for (int c = 0; c < nc; c++) {
            float s = buf[(4 + c) * total_anchors + i];
            if (s > max_s) max_s = s;
        }
        if (max_s >= thresh) candidates.push_back({i, max_s});
    }
    // 按 score 降序排序
    std::sort(candidates.begin(), candidates.end(),
              [](const AnchorScore& a, const AnchorScore& b) { return a.score > b.score; });
    std::vector<int> indices;
    int n = std::min(top_n, (int)candidates.size());
    for (int i = 0; i < n; i++) {
        indices.push_back(candidates[i].idx);
    }
    // 按索引升序排列，方便对比
    std::sort(indices.begin(), indices.end());
    return indices;
}

/// 将指定 anchor 索引的原始输出写入文本文件，供与 ONNX 对比
static void dump_raw_anchors(const float* buf, int attrs, int total_anchors,
                              const std::vector<int>& indices, int nc,
                              const char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "无法写入 %s\n", path); return; }
    fprintf(fp, "# RKNN raw output dump (high-confidence anchors only)\n");
    fprintf(fp, "# shape: [1, %d, %d]\n", attrs, total_anchors);
    fprintf(fp, "# layout: data[c * total_anchors + anchor_idx]\n");
    fprintf(fp, "# channels: cx(0) cy(1) w(2) h(3) cls0..cls%d(4..%d) angle(%d)\n",
            nc - 1, 4 + nc - 1, attrs - 1);
    fprintf(fp, "# dumped_indices:");
    for (int idx : indices) fprintf(fp, " %d", idx);
    fprintf(fp, "\n\n");

    for (int idx : indices) {
        if (idx < 0 || idx >= total_anchors) continue;
        // 找出最大类别
        int best_cls = 0;
        float best_score = buf[4 * total_anchors + idx];
        for (int c = 1; c < nc; c++) {
            float s = buf[(4 + c) * total_anchors + idx];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        fprintf(fp, "=== anchor[%d] === best_cls=%d score=%.6f\n", idx, best_cls, best_score);
        // 坐标通道
        fprintf(fp, "  cx=%.6f  cy=%.6f  w=%.6f  h=%.6f\n",
                buf[0 * total_anchors + idx],
                buf[1 * total_anchors + idx],
                buf[2 * total_anchors + idx],
                buf[3 * total_anchors + idx]);
        // 类别分数
        fprintf(fp, "  cls_probs:");
        for (int c = 0; c < nc; c++) {
            fprintf(fp, " %.6f", buf[(4 + c) * total_anchors + idx]);
        }
        fprintf(fp, "\n");
        // 角度
        fprintf(fp, "  angle=%.6f\n", buf[(attrs - 1) * total_anchors + idx]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("  \xE2\x9C\x93 raw output 已保存到 %s (%d 个高置信度 anchor)\n", path, (int)indices.size());
}

// ======================== Pimpl 实现 ========================

struct YoloRKNN::Impl {
#ifndef RKNN_STUB
    rknn_context ctx = 0;
#endif
    int model_imgsz = 640;
    int output_attrs = 31;     // 4 + 26 + 1
    int n_output   = 4;        // P2/P3/P4/P5 四个输出
    int layer_anchors[4] = {};  // 每层 anchor 数
    int total_anchors = 34000;

    // 拼接后的输出缓冲区
    std::vector<float> output_buf;
    // 每层独立缓冲区
    std::vector<std::vector<float>> layer_bufs;

    // dump 控制：前 N 帧输出原始 tensor 到文件
    int dump_count = 1;  // 只 dump 第 1 帧
    const char* dump_path = "rknn_raw_output.txt";
};

YoloRKNN::YoloRKNN() : impl_(std::make_unique<Impl>()) {}

YoloRKNN::~YoloRKNN() {
    release();
}

int YoloRKNN::load(const std::string& model_path,
                    const DecodeConfig& config,
                    const std::vector<std::string>& names) {
    config_ = config;
    names_ = names;

#ifdef RKNN_STUB
    fprintf(stderr, "错误: RKNN_STUB 模式，无法加载模型\n");
    return -1;
#else
    // 1. 初始化 RKNN 上下文
    int ret = rknn_init(&impl_->ctx, (void*)model_path.c_str(),
                         0, 0, nullptr);  // size=0 表示 model 是文件路径
    if (ret != 0) {
        fprintf(stderr, "错误: rknn_init 失败 (ret=%d): %s\n", ret, model_path.c_str());
        return ret;
    }

    // 2. 查询模型输入输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        fprintf(stderr, "错误: rknn_query 失败 (ret=%d)\n", ret);
        return ret;
    }
    impl_->n_output = io_num.n_output;
    printf("  模型输入数: %d, 输出数: %d\n", io_num.n_input, io_num.n_output);

    // 3. 查询每个输出 tensor 的 shape，计算每层 anchor 数
    impl_->total_anchors = 0;
    impl_->layer_bufs.resize(io_num.n_output);
    for (int i = 0; i < (int)io_num.n_output; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != 0) {
            fprintf(stderr, "错误: 查询输出[%d] 失败 (ret=%d)\n", i, ret);
            return ret;
        }
        printf("  输出[%d] dims: [", i);
        for (int d = 0; d < (int)attr.n_dims; d++) {
            printf("%d%s", attr.dims[d], d < (int)attr.n_dims - 1 ? ", " : "");
        }
        printf("]\n");

        // 期望 shape: [1, attrs, anchors_i] 或 [1, attrs, H, W]
        int anchors_i = 0;
        if (attr.n_dims == 3) {
            impl_->output_attrs = attr.dims[1];
            anchors_i = attr.dims[2];
        } else if (attr.n_dims == 4) {
            impl_->output_attrs = attr.dims[1];
            anchors_i = attr.dims[2] * attr.dims[3];  // H * W
        }
        if (i < 4) {
            impl_->layer_anchors[i] = anchors_i;
        }
        impl_->total_anchors += anchors_i;
        impl_->layer_bufs[i].resize(impl_->output_attrs * anchors_i);
    }
    printf("  总 anchor 数: %d, attrs: %d\n", impl_->total_anchors, impl_->output_attrs);

    // 预分配拼接缓冲区
    impl_->output_buf.resize(impl_->output_attrs * impl_->total_anchors);

    loaded_ = true;
    printf("  ✓ RKNN 模型加载成功\n");
    return 0;
#endif
}

std::vector<Detection> YoloRKNN::detect(const cv::Mat& img) {
    if (!loaded_) {
        fprintf(stderr, "错误: 模型未加载\n");
        return {};
    }

    // ======================== 1. Letterbox 预处理 ========================
    std::vector<uint8_t> input_data;
    LetterboxInfo lb = preprocess_letterbox(img, input_data,
                                             impl_->model_imgsz, impl_->model_imgsz);

#ifdef RKNN_STUB
    return {};
#else
    // ======================== 2. 设置输入 ========================
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;    // 原始像素 [0, 255]
    inputs[0].fmt = RKNN_TENSOR_NHWC;      // NHWC 格式
    inputs[0].size = input_data.size();
    inputs[0].buf = input_data.data();
    inputs[0].pass_through = 0;  // 让 RKNN 做 mean/std 归一化

    int ret = rknn_inputs_set(impl_->ctx, 1, inputs);
    if (ret != 0) {
        fprintf(stderr, "错误: rknn_inputs_set 失败 (ret=%d)\n", ret);
        return {};
    }

    // ======================== 3. 推理 ========================
    int n_out = impl_->n_output;
    std::vector<rknn_output> outputs(n_out);
    memset(outputs.data(), 0, sizeof(rknn_output) * n_out);
    for (int i = 0; i < n_out; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;       // 请求 float32 输出
        outputs[i].is_prealloc = 1;
        outputs[i].buf = impl_->layer_bufs[i].data();
        outputs[i].size = impl_->layer_bufs[i].size() * sizeof(float);
    }

    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != 0) {
        fprintf(stderr, "错误: rknn_run 失败 (ret=%d)\n", ret);
        return {};
    }

    ret = rknn_outputs_get(impl_->ctx, n_out, outputs.data(), nullptr);
    if (ret != 0) {
        fprintf(stderr, "错误: rknn_outputs_get 失败 (ret=%d)\n", ret);
        return {};
    }

    // ======================== 4. 拼接 4 个 tensor ========================
    // 每个 tensor shape: [1, attrs, anchors_i]
    // 拼接后: [1, attrs, total_anchors]（按 anchor 维度拼接）
    int attrs = impl_->output_attrs;
    float* dst = impl_->output_buf.data();
    int anchor_offset = 0;
    for (int i = 0; i < n_out; i++) {
        int n = impl_->layer_anchors[i];
        const float* src = impl_->layer_bufs[i].data();
        // 按通道复制：每个通道连续 n 个元素
        for (int c = 0; c < attrs; c++) {
            memcpy(dst + c * impl_->total_anchors + anchor_offset,
                   src + c * n,
                   n * sizeof(float));
        }
        anchor_offset += n;
    }

    // ======================== 4.5 dump 原始输出（前 N 帧）========================
    if (impl_->dump_count > 0) {
        impl_->dump_count--;
        // 自动找出高置信度 anchor (cls_score > 0.01, 取 top 20)
        int nc = attrs - 5;  // 26 类
        auto dump_indices = find_top_anchors(impl_->output_buf.data(), attrs,
                                              impl_->total_anchors, nc,
                                              0.01f, 20);
        printf("  [dump] 找到 %d 个高置信度 anchor\n", (int)dump_indices.size());
        if (!dump_indices.empty()) {
            dump_raw_anchors(impl_->output_buf.data(), attrs,
                             impl_->total_anchors, dump_indices, nc,
                             impl_->dump_path);
        }
    }

    // ======================== 5. 解码 ========================
    auto dets = decode_mode_c(
        impl_->output_buf.data(),
        impl_->output_attrs,
        impl_->total_anchors,
        config_, names_,
        lb.ratio, lb.pad_x, lb.pad_y);

    // 释放输出
    rknn_outputs_release(impl_->ctx, n_out, outputs.data());

    return dets;
#endif
}

void YoloRKNN::release() {
    if (!loaded_) return;
#ifndef RKNN_STUB
    rknn_destroy(impl_->ctx);
    impl_->ctx = 0;
#endif
    loaded_ = false;
}

}  // namespace yolo_rknn
