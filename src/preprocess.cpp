/**
 * @file preprocess.cpp
 * @brief Letterbox 图像预处理实现
 */

#include "preprocess.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace yolo_rknn {

LetterboxInfo preprocess_letterbox(const cv::Mat& src,
                                    std::vector<uint8_t>& dst,
                                    int target_w,
                                    int target_h) {
    int h0 = src.rows;
    int w0 = src.cols;

    // 计算缩放比: 取宽高方向较小的比例，确保图像完整放入目标尺寸
    float ratio = std::min(static_cast<float>(target_w) / w0,
                           static_cast<float>(target_h) / h0);
    int new_w = static_cast<int>(w0 * ratio);
    int new_h = static_cast<int>(h0 * ratio);

    // 居中填充偏移
    float pad_x = (target_w - new_w) / 2.0f;
    float pad_y = (target_h - new_h) / 2.0f;

    // 等比缩放
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // BGR → 灰度 → 3通道灰度图（灰度模型需要单通道信息复制到3通道）
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray3;
    cv::cvtColor(gray, gray3, cv::COLOR_GRAY2RGB);

    // 创建灰边(114)画布
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));

    // 将缩放后图像粘贴到画布中心
    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    gray3.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));

    // 输出 NHWC uint8 (直接取 canvas 的连续数据)
    dst.resize(target_h * target_w * 3);
    std::memcpy(dst.data(), canvas.data, dst.size());

    return {ratio, pad_x, pad_y};
}

}  // namespace yolo_rknn
