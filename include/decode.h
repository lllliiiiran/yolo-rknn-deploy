#pragma once
/**
 * @file decode.h
 * @brief YOLO11-OBB 后处理解码
 *
 * Mode C 单 tensor [1, 31, 34000]: 已解码格式 (xywh + cls概率 + angle)
 *
 * 后处理流程:
 *   1. 单一阈值过滤 (cls 已是概率，不做 sigmoid)
 *   2. 按 score 排序
 *   3. Letterbox 反算 → 原图坐标
 *   4. OBB NMS (AABB 预筛 + 旋转 IoU)
 *   5. top-k 截断
 */

#include <string>
#include <vector>
#include <cmath>

namespace yolo_rknn {

/// 单个检测结果
struct Detection {
    float score;        ///< 置信度
    int class_id;       ///< 类别 ID
    std::string class_name;  ///< 类别名称
    float x, y;         ///< 中心点 (原图坐标)
    float w, h;         ///< 宽高 (原图坐标)
    float angle;        ///< 旋转角 (弧度, [-π/4, 3π/4])
};

/// 解码配置
struct DecodeConfig {
    int imgsz = 640;                     ///< 模型输入尺寸
    int num_classes = 26;                ///< 类别数
    float default_conf = 0.25f;          ///< 默认置信度阈值

    // 分层置信度阈值 (P2/P3/P4/P5)
    float layer_conf[4] = {0.65f, 0.30f, 0.25f, 0.20f};
    int strides[4] = {4, 8, 16, 32};     ///< 各层步长
    int start_layer = 0;                 ///< 起始层（0=P2, 1=P3, 2=P4, 3=P5），跳过前几层以提升速度

    // ── 每层尺寸范围（letterbox 像素，按 max(w,h) 判断）──
    // 各层范围互不重叠，避免同一目标被多层重复检测
    float layer_size_min[4] = {4.f,   32.f,   80.f,  192.f};   ///< P2/P3/P4/P5 最小尺寸
    float layer_size_max[4] = {32.f,  80.f,  192.f,  640.f};   ///< P2/P3/P4/P5 最大尺寸
    
    int topk = 1000;                      ///< 每张图最大检测数
    int class_topk = 11;                 ///< 每个类别保留的最大 anchor 数（逐类别筛选）
    float nms_iou = 0.30f;               ///< OBB NMS IoU 阈値（OBB角度敏感，用较小値）

    // 面积约束 (像素²)
    float min_area = 50.0f;              ///< 最小面积
    float max_area = 400000.0f;          ///< 最大面积
    float max_aspect_ratio = 10.0f;      ///< 最大宽高比

    bool use_nms = true;                 ///< 是否启用 OBB NMS
};

/**
 * @brief 解码 Mode C 格式输出 [1, 4+K+1, N]
 *
 * 输出布局: x(1) y(1) w(1) h(1) + cls_probs(K) + angle(1)
 * 注意: cls 已经是概率 [0,1]，不需要再做 sigmoid！
 *
 * @param output    模型输出数据 (float32, 行优先)
 * @param config    解码配置
 * @param names     类别名称列表
 * @param lb_ratio  letterbox 缩放比
 * @param lb_pad_x  letterbox 水平偏移
 * @param lb_pad_y  letterbox 垂直偏移
 * @return 检测结果列表 (原图坐标)
 */
std::vector<Detection> decode_mode_c(
    const float* output,
    int attrs,           // 属性维度 (4 + nc + 1 = 31)
    int num_anchors,     // anchor 总数 (34000)
    const DecodeConfig& config,
    const std::vector<std::string>& names,
    float lb_ratio, float lb_pad_x, float lb_pad_y);

/**
 * @brief OBB NMS (非极大值抑制)
 *
 * 算法:
 *   1. 按 score 降序排序
 *   2. 按类别分组
 *   3. AABB 预筛选 (快速排除不可能重叠的框)
 *   4. 精确旋转 IoU (Sutherland-Hodgman 多边形裁剪)
 *
 * @param dets      检测结果 (就地修改，抑制低分框)
 * @param iou_thres IoU 阈值
 */
void obb_nms(std::vector<Detection>& dets, float iou_thres);

/**
 * @brief Letterbox 坐标反算
 *
 * 正变换: model = orig * ratio + pad
 * 反变换: orig = (model - pad) / ratio
 * w/h 只需除以 ratio (pad 是平移，不影响尺寸)
 */
inline void det_to_orig(Detection& d, float ratio, float pad_x, float pad_y) {
    d.x = (d.x - pad_x) / ratio;
    d.y = (d.y - pad_y) / ratio;
    d.w /= ratio;
    d.h /= ratio;
    // angle 不受等比缩放影响
}

}  // namespace yolo_rknn
