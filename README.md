# YOLO11-OBB RKNN C++ 部署

在瑞芯微 RK3588/RK3566/RK3576 平台上部署 YOLO11-OBB 旋转框检测模型。

## 特性

- **分层置信度阈值**: P2 层阈值更高 (0.45)，减少冗余检测
- **OBB NMS**: AABB 预筛选 + Sutherland-Hodgman 旋转 IoU
- **面积/宽高比约束**: 过滤异常框
- **零拷贝优化**: 预分配输出缓冲区
- **JSON 输出**: 标准检测结果格式

## 编译

### 开发板 (RK3588/RK3566)

```bash
# 安装依赖
sudo apt install cmake g++ libopencv-dev

# 设置 RKNN SDK 路径
export RKNN_SDK_PATH=/path/to/rknpu2/runtime

# 编译
cd deploy
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### x86 交叉编译

```bash
# 使用 aarch64 交叉编译器
cmake .. -DCMAKE_TOOLCHAIN_FILE=../aarch64_toolchain.cmake
make -j4
```

## 使用

```bash
# 基础推理
./build/yolo_rknn \
  --model /path/to/best.rknn \
  --source /path/to/images \
  --out results

# 自定义阈值 + 可视化
./build/yolo_rknn \
  --model best.rknn \
  --source test_images/ \
  --out results \
  --p2-conf 0.50 \
  --p3-conf 0.30 \
  --p4-conf 0.25 \
  --p5-conf 0.20 \
  --nms-iou 0.45 \
  --visualize
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | best.rknn | RKNN 模型路径 |
| `--source` | data/images | 图片目录或单张图片 |
| `--out` | results | JSON + 可视化输出目录 |
| `--imgsz` | 640 | 模型输入尺寸 |
| `--conf` | 0.25 | 默认置信度阈值 |
| `--topk` | 100 | 每张图最大检测数 |
| `--nms-iou` | 0.45 | OBB NMS IoU 阈值 |
| `--visualize` | 关 | 保存 OBB 可视化图片 |
| `--max-images` | 0 | 最多处理图片数 (0=全部) |
| `--p2-conf` | 0.45 | P2 层阈值 (高分辨率层) |
| `--p3-conf` | 0.30 | P3 层阈值 |
| `--p4-conf` | 0.25 | P4 层阈值 |
| `--p5-conf` | 0.20 | P5 层阈值 (低分辨率层) |

## 输出格式

### JSON

```json
{
  "image": "test.jpg",
  "orig_size": [1920, 1080],
  "detections": [
    {
      "class": "front_head",
      "class_id": 7,
      "score": 0.92,
      "x": 960.0,
      "y": 540.0,
      "w": 200.0,
      "h": 180.0,
      "angle_rad": 0.02,
      "angle_deg": 1.15
    }
  ]
}
```

## 模型输出格式

- **Shape**: `[1, 31, 34000]` (640 输入，P2+P3+P4+P5 = 34000 anchors)
- **布局**: `x, y, w, h (4) + cls_probs (26) + angle (1)`
- **cls**: 已经是 sigmoid 概率 [0, 1]，**不需要再做 sigmoid**
- **angle**: 弧度制 [-π/4, 3π/4]

## 分层阈值策略

| 层 | Stride | Anchor 数 | 占比 | 推荐阈值 | 原因 |
|---|---|---|---|---|---|
| P2 | 4 | 25,600 | 75% | 0.45 | anchor 多且冗余，提高阈值减少误检 |
| P3 | 8 | 6,400 | 19% | 0.30 | 中等目标，平衡召回和精度 |
| P4 | 16 | 1,600 | 5% | 0.25 | 大目标，得分高且定位准 |
| P5 | 32 | 400 | 1% | 0.20 | 超大目标，低阈值确保不漏检 |

## 性能参考 (RK3588)

| 指标 | 数值 |
|---|---|
| 推理耗时 | ~30ms (NPU) |
| 后处理 | ~5ms (CPU) |
| 总延迟 | ~35ms/帧 |
| FPS | ~28 |
