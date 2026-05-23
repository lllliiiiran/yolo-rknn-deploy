/**
 * @file main.cpp
 * @brief YOLO11-OBB RKNN 推理入口
 *
 * 功能:
 *   - 批量推理目录下的图片
 *   - 输出 JSON 检测结果
 *   - 可选可视化 (OBB 旋转框)
 *
 * 用法:
 *   ./yolo_rknn --model best.rknn --source images/ --out results/ --visualize
 *   ./yolo_rknn --model best.rknn --source test.jpg --conf 0.3
 */

#include "yolo_rknn.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>

#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

// ======================== 工具函数 ========================

static std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        // 可能是单个文件
        files.push_back(dir);
        return files;
    }
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > 4) {
            std::string ext = name.substr(name.size() - 4);
            for (auto& c : ext) c = tolower(c);
            if (ext == ".jpg" || ext == "jpeg" || ext == ".png") {
                files.push_back(dir + "/" + name);
            }
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

static void mkdir_p(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

/// 简单的 JSON 序列化 (避免引入第三方库)
static std::string to_json(const std::string& image_name,
                            int w0, int h0,
                            const std::vector<yolo_rknn::Detection>& dets) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"image\": \"" << image_name << "\",\n";
    ss << "  \"orig_size\": [" << w0 << ", " << h0 << "],\n";
    ss << "  \"detections\": [\n";
    for (size_t i = 0; i < dets.size(); i++) {
        auto& d = dets[i];
        ss << "    {\n";
        ss << "      \"class\": \"" << d.class_name << "\",\n";
        ss << "      \"class_id\": " << d.class_id << ",\n";
        ss << "      \"score\": " << std::round(d.score * 10000) / 10000.0 << ",\n";
        ss << "      \"x\": " << std::round(d.x * 100) / 100.0 << ",\n";
        ss << "      \"y\": " << std::round(d.y * 100) / 100.0 << ",\n";
        ss << "      \"w\": " << std::round(d.w * 100) / 100.0 << ",\n";
        ss << "      \"h\": " << std::round(d.h * 100) / 100.0 << ",\n";
        ss << "      \"angle_rad\": " << std::round(d.angle * 10000) / 10000.0 << ",\n";
        ss << "      \"angle_deg\": " << std::round(d.angle * 180.0 / M_PI * 100) / 100.0 << "\n";
        ss << "    }" << (i < dets.size() - 1 ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
}

#ifdef ENABLE_VISUALIZE
/// 绘制 OBB 旋转检测框
static void draw_detections(cv::Mat& img,
                             const std::vector<yolo_rknn::Detection>& dets) {
    for (auto& d : dets) {
        float angle_deg = d.angle * 180.0f / M_PI;
        cv::RotatedRect rect(cv::Point2f(d.x, d.y),
                               cv::Size2f(d.w, d.h),
                               angle_deg);
        cv::Point2f pts[4];
        rect.points(pts);

        // 绘制旋转框 (橙色)
        for (int i = 0; i < 4; i++) {
            cv::line(img, pts[i], pts[(i + 1) % 4],
                     cv::Scalar(0, 128, 255), 2);
        }

        // 标注文字
        char label[128];
        snprintf(label, sizeof(label), "%s %.2f", d.class_name.c_str(), d.score);
        cv::putText(img, label, cv::Point(d.x, std::max(10.0f, d.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 128, 255), 1);
    }
}
#endif

// ======================== 类别名称 ========================

static std::vector<std::string> get_default_names() {
    return {
        "eye", "eye_close", "eye_open", "mouth", "nose",
        "ear", "forehead", "front_head", "side_head", "top_head",
        "label11", "baby_hand", "baby_chin", "baby_chest", "baby_foot",
        "baby_leg", "baby_full_body", "baby_coverd_full_body",
        "baby_hat", "baby_blinder", "baby_arm",
        "parents_hand", "parents_body",
        "supine", "prone", "lateral"
    };
}

// ======================== 参数解析 ========================

struct Args {
    std::string model = "best.rknn";
    std::string source = "data/images";
    std::string out = "results";
    int imgsz = 640;
    float conf = 0.25f;
    int topk = 1000;
    float nms_iou = 0.30f;
    bool visualize = false;
    int max_images = 0;

    // 分层阈值
    float p2_conf = 0.45f;
    float p3_conf = 0.30f;
    float p4_conf = 0.25f;
    float p5_conf = 0.20f;

    void print() const {
        printf("=== 推理参数 ===\n");
        printf("  模型:     %s\n", model.c_str());
        printf("  输入:     %s\n", source.c_str());
        printf("  输出:     %s\n", out.c_str());
        printf("  输入尺寸: %d\n", imgsz);
        printf("  阈值:     P2=%.2f P3=%.2f P4=%.2f P5=%.2f\n",
               p2_conf, p3_conf, p4_conf, p5_conf);
        printf("  NMS IoU:  %.2f\n", nms_iou);
        printf("  Top-K:    %d\n", topk);
        printf("  可视化:   %s\n", visualize ? "开" : "关");
    }
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (arg == "--model" || arg == "-m") args.model = next();
        else if (arg == "--source" || arg == "-s") args.source = next();
        else if (arg == "--out" || arg == "-o") args.out = next();
        else if (arg == "--imgsz") args.imgsz = std::stoi(next());
        else if (arg == "--conf") args.conf = std::stof(next());
        else if (arg == "--topk") args.topk = std::stoi(next());
        else if (arg == "--nms-iou") args.nms_iou = std::stof(next());
        else if (arg == "--visualize") args.visualize = true;
        else if (arg == "--max-images") args.max_images = std::stoi(next());
        else if (arg == "--p2-conf") args.p2_conf = std::stof(next());
        else if (arg == "--p3-conf") args.p3_conf = std::stof(next());
        else if (arg == "--p4-conf") args.p4_conf = std::stof(next());
        else if (arg == "--p5-conf") args.p5_conf = std::stof(next());
        else if (arg == "--help" || arg == "-h") {
            printf("用法: %s [选项]\n\n", argv[0]);
            printf("选项:\n");
            printf("  --model, -m     RKNN 模型路径 (默认: best.rknn)\n");
            printf("  --source, -s    图片目录或单张图片 (默认: data/images)\n");
            printf("  --out, -o       输出目录 (默认: results)\n");
            printf("  --imgsz         模型输入尺寸 (默认: 640)\n");
            printf("  --conf          默认置信度阈值 (默认: 0.25)\n");
            printf("  --topk          最大检测数 (默认: 100)\n");
            printf("  --nms-iou       OBB NMS IoU 阈值 (默认: 0.45)\n");
            printf("  --visualize     保存可视化图片\n");
            printf("  --max-images    最多处理图片数 (0=全部)\n");
            printf("  --p2-conf       P2 层阈值 (默认: 0.45)\n");
            printf("  --p3-conf       P3 层阈值 (默认: 0.30)\n");
            printf("  --p4-conf       P4 层阈值 (默认: 0.25)\n");
            printf("  --p5-conf       P5 层阈值 (默认: 0.20)\n");
            exit(0);
        }
    }
    return args;
}

// ======================== 主程序 ========================

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    args.print();

    // 配置
    yolo_rknn::DecodeConfig config;
    config.imgsz = args.imgsz;
    config.num_classes = 26;
    config.default_conf = args.conf;
    config.layer_conf[0] = args.p2_conf;
    config.layer_conf[1] = args.p3_conf;
    config.layer_conf[2] = args.p4_conf;
    config.layer_conf[3] = args.p5_conf;
    config.topk = args.topk;
    config.nms_iou = args.nms_iou;

    // 加载模型
    printf("\n==> 加载模型...\n");
    yolo_rknn::YoloRKNN detector;
    auto names = get_default_names();
    if (detector.load(args.model, config, names) != 0) {
        return 1;
    }

    // 收集图片
    auto files = list_images(args.source);
    if (args.max_images > 0 && (int)files.size() > args.max_images) {
        files.resize(args.max_images);
    }
    printf("\n==> 图片数: %zu\n", files.size());

    // 创建输出目录
    mkdir_p(args.out);
    if (args.visualize) mkdir_p(args.out + "/vis");

    // 逐张推理
    int total_dets = 0;
    double total_ms = 0;

    for (size_t idx = 0; idx < files.size(); idx++) {
        auto& img_path = files[idx];
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            fprintf(stderr, "  跳过: 无法读取 %s\n", img_path.c_str());
            continue;
        }

        // 提取文件名
        std::string basename = img_path;
        auto slash = basename.rfind('/');
        if (slash != std::string::npos) basename = basename.substr(slash + 1);
        auto dot = basename.rfind('.');
        std::string stem = (dot != std::string::npos) ? basename.substr(0, dot) : basename;

        printf("\n============================================================\n");
        printf("[%zu/%zu] %s\n", idx + 1, files.size(), basename.c_str());
        printf("============================================================\n");

        // 计时推理
        auto t0 = std::chrono::high_resolution_clock::now();
        auto dets = detector.detect(img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;

        printf("  耗时: %.1f ms\n", ms);
        printf("  检测数: %zu\n", dets.size());
        total_dets += dets.size();

        // 打印 top-5
        for (size_t i = 0; i < std::min(dets.size(), (size_t)5); i++) {
            auto& d = dets[i];
            printf("    [%zu] %-20s score=%.3f  xywh=(%.0f,%.0f,%.0f,%.0f)  angle=%.1f°\n",
                   i, d.class_name.c_str(), d.score,
                   d.x, d.y, d.w, d.h,
                   d.angle * 180.0 / M_PI);
        }

        // 保存 JSON
        std::string json = to_json(basename, img.cols, img.rows, dets);
        std::string json_path = args.out + "/" + stem + ".json";
        std::ofstream(json_path) << json;

        // 保存可视化
#ifdef ENABLE_VISUALIZE
        if (args.visualize) {
            cv::Mat vis = img.clone();
            draw_detections(vis, dets);
            cv::imwrite(args.out + "/vis/" + stem + ".jpg", vis);
        }
#endif
    }

    // 汇总
    printf("\n============================================================\n");
    printf("推理完成\n");
    printf("============================================================\n");
    printf("  总图片:   %zu\n", files.size());
    printf("  总检测数: %d\n", total_dets);
    printf("  总耗时:   %.1f ms\n", total_ms);
    if (!files.empty()) {
        printf("  平均耗时: %.1f ms/张\n", total_ms / files.size());
    }
    printf("  输出目录: %s\n", args.out.c_str());

    detector.release();
    return 0;
}
