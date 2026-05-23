#pragma once
/**
 * @file yolo_rknn.h
 * @brief YOLO11-OBB RKNN 推理接口
 *
 * 封装 RKNN 模型加载、推理、后处理的完整流程。
 * 支持 RK3588/RK3566/RK3576 等瑞芯微平台。
 *
 * 使用示例:
 *   YoloRKNN detector;
 *   detector.load("best.rknn");
 *   auto dets = detector.detect(image);
 *   for (auto& d : dets) { ... }
 */

#include "decode.h"
#include "preprocess.h"
#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace yolo_rknn {

/// RKNN 推理器
class YoloRKNN {
public:
    YoloRKNN();
    ~YoloRKNN();

    // 禁止拷贝
    YoloRKNN(const YoloRKNN&) = delete;
    YoloRKNN& operator=(const YoloRKNN&) = delete;

    /**
     * @brief 加载 RKNN 模型
     * @param model_path  .rknn 模型文件路径
     * @param config      解码配置
     * @param names       类别名称列表
     * @return 0=成功, 非0=失败
     */
    int load(const std::string& model_path,
             const DecodeConfig& config = {},
             const std::vector<std::string>& names = {});

    /**
     * @brief 检测单张图像
     * @param img  输入图像 (BGR, OpenCV 格式)
     * @return 检测结果列表 (原图坐标)
     */
    std::vector<Detection> detect(const cv::Mat& img);

    /**
     * @brief 释放 RKNN 资源
     */
    void release();

    /// 获取模型是否已加载
    bool is_loaded() const { return loaded_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
    DecodeConfig config_;
    std::vector<std::string> names_;
};

}  // namespace yolo_rknn
