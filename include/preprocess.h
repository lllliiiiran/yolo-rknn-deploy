#pragma once
/**
 * @file preprocess.h
 * @brief Letterbox 图像预处理
 *
 * 保持宽高比缩放 + 灰边(114)居中填充，与 YOLO 训练时一致。
 * 输出 NHWC uint8 格式，供 RKNN 推理（RKNN 内部做 /255 归一化）。
 */

#include <opencv2/core.hpp>
#include <vector>

namespace yolo_rknn {

/// Letterbox 参数，用于坐标反算
struct LetterboxInfo {
    float ratio;    ///< 缩放比例 = min(target_w/orig_w, target_h/orig_h)
    float pad_x;    ///< 水平填充偏移
    float pad_y;    ///< 垂直填充偏移
};

/**
 * @brief Letterbox 预处理
 *
 * 流程:
 *   1. BGR → RGB
 *   2. 计算 ratio = min(target_w/w0, target_h/h0)
 *   3. 等比缩放
 *   4. 灰边(114)居中填充到目标尺寸
 *   5. 输出 NHWC uint8 [0, 255]
 *
 * @param src       输入图像 (BGR)
 * @param dst       输出数据 (NHWC uint8, 大小为 target_h × target_w × 3)
 * @param target_w  目标宽度
 * @param target_h  目标高度
 * @return LetterboxInfo 用于坐标反算
 */
LetterboxInfo preprocess_letterbox(const cv::Mat& src,
                                    std::vector<uint8_t>& dst,
                                    int target_w = 640,
                                    int target_h = 640);

}  // namespace yolo_rknn
