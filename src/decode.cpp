/**
 * @file decode.cpp
 * @brief YOLO11-OBB 后处理解码实现
 *
 * 后处理流程:
 *   1. 按层遍历 anchor，分层阈值过滤 (P2~P5)
 *   2. 按 score 排序
 *   3. Letterbox 反算 → 原图坐标
 *   4. OBB NMS: AABB 预筛 + Sutherland-Hodgman 旋转 IoU
 *   5. top-k 截断
 *
 * 注意: 图像坐标系 y 轴向下，旋转顶点用 CW 顺序，
 *       Sutherland-Hodgman cross >= 0 才表示 "inside"
 */

#include "decode.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace yolo_rknn {

// ======================== Mode C 解码 ========================

std::vector<Detection> decode_mode_c(
    const float* output,
    int attrs,
    int num_anchors,
    const DecodeConfig& config,
    const std::vector<std::string>& names,
    float lb_ratio, float lb_pad_x, float lb_pad_y)
{
    int nc = config.num_classes;
    int class_topk = config.class_topk;

    // 计算有效 anchor 范围（跳过 start_layer 以前的层）
    int anchor_offset = 0;
    for (int layer = 0; layer < config.start_layer; layer++) {
        int grid = config.imgsz / config.strides[layer];
        anchor_offset += grid * grid;
    }
    int total_valid = num_anchors - anchor_offset;  // 有效 anchor 数

    // 构建每个 anchor 对应的分层阈值
    std::vector<float> anchor_thresh(total_valid, 0.0f);
    {
        int off = 0;
        for (int layer = config.start_layer; layer < 4; layer++) {
            int grid = config.imgsz / config.strides[layer];
            int n = grid * grid;
            float th = config.layer_conf[layer];
            for (int i = 0; i < n && (off + i) < total_valid; i++) {
                anchor_thresh[off + i] = th;
            }
            off += n;
        }
    }

    // 逐类别 topk 筛选: 每个类别保留得分最高的 class_topk 个 anchor
    // anchor_selected[idx] = {best_class, best_score}，同一 anchor 被多类选中时保留最高分
    struct Candidate {
        int class_id;
        float score;
    };
    std::vector<Candidate> anchor_selected(total_valid, {-1, 0.0f});

    for (int c = 0; c < nc; c++) {
        // 收集该类别超过阈值的 anchor
        struct AnchorScore { int idx; float score; };
        std::vector<AnchorScore> candidates;
        candidates.reserve(512);

        for (int i = 0; i < total_valid; i++) {
            float s = output[(4 + c) * num_anchors + (anchor_offset + i)];
            if (s >= anchor_thresh[i]) {
                candidates.push_back({i, s});
            }
        }

        // 按得分降序排序，取 top class_topk
        std::sort(candidates.begin(), candidates.end(),
                  [](const AnchorScore& a, const AnchorScore& b) {
                      return a.score > b.score;
                  });
        int take = std::min(class_topk, (int)candidates.size());

        for (int k = 0; k < take; k++) {
            int idx = candidates[k].idx;
            float s = candidates[k].score;
            // 同一 anchor 被多类选中时，保留最高分的类别
            if (s > anchor_selected[idx].score) {
                anchor_selected[idx] = {c, s};
            }
        }
    }

    // 构建检测列表
    std::vector<Detection> dets;
    dets.reserve(256);

    for (int i = 0; i < total_valid; i++) {
        if (anchor_selected[i].class_id < 0) continue;  // 未被任何类别选中

        int idx = anchor_offset + i;
        int best_cls = anchor_selected[i].class_id;
        float best_score = anchor_selected[i].score;

        float x = output[0 * num_anchors + idx];
        float y = output[1 * num_anchors + idx];
        float w = output[2 * num_anchors + idx];
        float h = output[3 * num_anchors + idx];
        float angle = output[(4 + nc) * num_anchors + idx];

        Detection det;
        det.score = best_score;
        det.class_id = best_cls;
        det.class_name = (best_cls < (int)names.size()) ? names[best_cls]
                                                        : std::to_string(best_cls);
        det.x = x;
        det.y = y;
        det.w = w;
        det.h = h;
        det.angle = angle;

        dets.push_back(det);
    }

    // 按 score 降序排序
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    // 调试: 输出候选框统计
    if (!dets.empty()) {
        float min_score = dets.back().score;
        float max_score = dets.front().score;
        int below_25 = 0;
        for (auto& d : dets) if (d.score < 0.25f) below_25++;
        fprintf(stderr, "  [decode] 候选框: %d个, 分数 [%.3f, %.3f], <0.25有%d个\n",
                (int)dets.size(), min_score, max_score, below_25);
    }

    // Letterbox 反算 → 原图坐标
    for (auto& det : dets) {
        det_to_orig(det, lb_ratio, lb_pad_x, lb_pad_y);
    }

    // OBB NMS（在原图空间，先去除重叠框）
    if (config.use_nms && !dets.empty()) {
        int before = (int)dets.size();
        obb_nms(dets, config.nms_iou);
        fprintf(stderr, "  [NMS] %d -> %d (抑制 %d 个)\n", before, (int)dets.size(), before - (int)dets.size());
    }

    // top-k 截断（NMS 后再截断，避免重叠框占满名额）
    if ((int)dets.size() > config.topk) {
        dets.resize(config.topk);
    }

    return dets;
}

// ======================== OBB NMS ========================

/// 轴对齐包围框
struct AABB { float x1, y1, x2, y2; };

/// 快速判断两个 AABB 是否相交（比计算 IoU 少 4 次乘除）
static inline bool aabb_overlap(const AABB& a, const AABB& b) {
    return a.x1 < b.x2 && a.x2 > b.x1 &&
           a.y1 < b.y2 && a.y2 > b.y1;
}

/// 获取旋转矩形的 4 个顶点（CW 顺时针绕序，图像坐标系 y 轴向下）
/// 注意: 数学 CCW 在图像坐标系(y↓)中变为 CW，
/// Sutherland-Hodgman 的 cross >= 0 inside 测试需要 CW 绕序
static void get_rotated_rect_points(const Detection& d, float pts[4][2]) {
    float cx = d.x, cy = d.y;
    float hw = d.w / 2.0f, hh = d.h / 2.0f;
    float cos_a = std::cos(d.angle);
    float sin_a = std::sin(d.angle);

    // 4 个角点（CW 顺时针，图像坐标系）: (-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)
    float dx[4] = {-hw,  hw,  hw, -hw};
    float dy[4] = {-hh, -hh,  hh,  hh};
    for (int i = 0; i < 4; i++) {
        pts[i][0] = cx + dx[i] * cos_a - dy[i] * sin_a;
        pts[i][1] = cy + dx[i] * sin_a + dy[i] * cos_a;
    }
}

/// 从旋转矩形顶点计算紧测 AABB（比外接圆更精确）
static AABB pts_to_aabb(const float pts[4][2]) {
    float x1 = pts[0][0], x2 = pts[0][0];
    float y1 = pts[0][1], y2 = pts[0][1];
    for (int i = 1; i < 4; i++) {
        if (pts[i][0] < x1) x1 = pts[i][0];
        if (pts[i][0] > x2) x2 = pts[i][0];
        if (pts[i][1] < y1) y1 = pts[i][1];
        if (pts[i][1] > y2) y2 = pts[i][1];
    }
    return {x1, y1, x2, y2};
}

/// 叉积
static float cross(float ox, float oy, float ax, float ay, float bx, float by) {
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

/// 多边形面积 (Shoelace formula)
static float polygon_area(const float pts[][2], int n) {
    float area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += pts[i][0] * pts[j][1];
        area -= pts[j][0] * pts[i][1];
    }
    return std::abs(area) / 2.0f;
}

/// Sutherland-Hodgman 多边形裁剪 (求两个凸多边形的交集)
static int clip_polygon(const float subject[][2], int sn,
                         const float clip[][2], int cn,
                         float output[][2]) {
    float temp[8][2];
    int tn = sn;
    std::memcpy(temp, subject, sn * sizeof(float) * 2);

    for (int i = 0; i < cn; i++) {
        int j = (i + 1) % cn;

        float result[8][2];
        int rn = 0;

        for (int k = 0; k < tn; k++) {
            int l = (k + 1) % tn;
            float d1 = cross(clip[i][0], clip[i][1],
                             clip[j][0], clip[j][1],
                             temp[k][0], temp[k][1]);
            float d2 = cross(clip[i][0], clip[i][1],
                             clip[j][0], clip[j][1],
                             temp[l][0], temp[l][1]);

            if (d1 >= 0 && rn < 8) { result[rn][0] = temp[k][0]; result[rn][1] = temp[k][1]; rn++; }
            if ((d1 >= 0) != (d2 >= 0) && rn < 8) {
                float t = d1 / (d1 - d2);
                result[rn][0] = temp[k][0] + t * (temp[l][0] - temp[k][0]);
                result[rn][1] = temp[k][1] + t * (temp[l][1] - temp[k][1]);
                rn++;
            }
        }
        tn = rn;
        std::memcpy(temp, result, rn * sizeof(float) * 2);
        if (tn == 0) break;
    }

    std::memcpy(output, temp, tn * sizeof(float) * 2);
    return tn;
}

void obb_nms(std::vector<Detection>& dets, float iou_thres) {
    if (dets.empty()) return;

    // 按 score 降序排序
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    const int n = (int)dets.size();

    // 预计算旋转顶点和 AABB，避免内层循环重复计算 cos/sin/sqrt
    struct BoxCache {
        float pts[4][2];  // 旋转矩形顶点（CW，图像坐标系）
        AABB  aabb;       // 紧测 AABB（从顶点计算）
        float area;       // 实际面积（= w * h）
    };
    std::vector<BoxCache> cache(n);
    for (int i = 0; i < n; i++) {
        get_rotated_rect_points(dets[i], cache[i].pts);
        cache[i].aabb = pts_to_aabb(cache[i].pts);
        cache[i].area = dets[i].w * dets[i].h;
    }

    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;
        const BoxCache& ci = cache[i];

        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) continue;
            // 按类别分组：不同类别互不影响
            if (dets[i].class_id != dets[j].class_id) continue;

            const BoxCache& cj = cache[j];

            // AABB 快速预筛（仅 4 次比较）
            if (!aabb_overlap(ci.aabb, cj.aabb)) continue;

            // 精确旋转 IoU（使用预计算顶点）
            float inter_pts[8][2];
            int inter_n = clip_polygon(ci.pts, 4, cj.pts, 4, inter_pts);
            if (inter_n < 3) continue;
            float inter_area = polygon_area(inter_pts, inter_n);
            float uni = ci.area + cj.area - inter_area;
            float iou = (uni > 0) ? inter_area / uni : 0.0f;

            if (iou > iou_thres) {
                suppressed[j] = true;
            }
        }
    }

    // 移除被抑制的框
    std::vector<Detection> kept;
    kept.reserve(n);
    for (int i = 0; i < n; i++) {
        if (!suppressed[i]) kept.push_back(dets[i]);
    }
    dets = std::move(kept);
}

}  // namespace yolo_rknn
