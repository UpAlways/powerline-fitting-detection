# 输电线路金具检测

作者：冯腾杰，华北电力大学（保定）。

本项目包含五种检测模型的训练与测试，以及使用固定 YOLOv8s 权重的切片、高分辨率推理对照。数据来自华北电力大学课题组，不公开图像、标注及划分清单；复算论文指标需获得数据与固定划分的访问授权。

## 文件用途

| 目录或文件 | 内容 |
| --- | --- |
| `scripts/` | E1—E5 基线、E7 切片的运行代码及数据工具 |
| `scripts_e8/` | E8 整图 1280 补充实验代码，保持服务器目录名 |
| `configs/` | 五模型真实训练参数存档；不是直接加载的配置入口 |
| `models/pretrained/` | 官方初始化权重的下载清单、校验工具；不含权重本体 |
| `results/` | 五模型测试表、分开的验证/测试推理对照表 |
| `weights-manifest.json` | Release 中五个训练后权重的文件名、大小和 SHA-256 |

下载本仓库即可获得代码和记录。训练后权重另从本仓库 Releases 获取 `trained-weights.zip`，解压到项目根目录后得到 `weights/*.pt` 或 `weights/*.pth`。官方初始化权重按 `models/pretrained/README.md` 准备。E7、E8 复用 E3 权重，没有独立训练权重。

模型结构由实际安装的 Torchvision 和 Ultralytics 提供，训练入口按模型类型调用对应网络。官方实现来源为 https://github.com/pytorch/vision 和 https://github.com/ultralytics/ultralytics 。本仓库交付实验脚本，不重复附带整套框架源码。

## 脚本版本与服务器目录

| 实验 | 服务器运行入口 | 本仓库复现入口 |
| --- | --- | --- |
| E1、E2 | `scripts/run_torchvision_baseline.py` | `scripts/run_torchvision_baseline.py` |
| E3–E5 | `scripts/run_val_baseline.py` | `scripts/run_val_baseline.py` |
| E7 | `scripts/run_sliced_baseline.py` | `scripts/run_sliced_baseline.py` |
| E8 | `scripts_e8/run_sliced_baseline.py` | `scripts_e8/run_sliced_baseline.py` |
| 训练集标注统计 | `scripts_e8/summarize_fittings_annotations.py` | `scripts_e8/summarize_fittings_annotations.py` |

`scripts/` 中前三个基线脚本取自基线归档，E7 推理脚本取自 E7 归档。E7 版固定输入 640，运行一次得到整图与整图加切片两组结果。`scripts_e8/` 使用当时为 E8 上传准备的本地版本，新增输入尺寸、仅整图模式、固定阈值诊断和尺寸分组 AP。两套目录各自带有 `evaluate_fittings.py`，运行入口会导入同目录的版本。`run_val_baseline.py` 在两套本地版本中内容相同，E8 从它导入哈希和 JSON 保存函数。`scripts/prepare_fittings.py` 是独立的数据转换工具。

## 运行环境和数据

原实验使用 Jetson Orin、Python 3.8.20、PyTorch 2.2.0、Torchvision 0.17.2 系列和 Ultralytics 8.3.221。还需 NumPy、Pillow、OpenCV 和 pycocotools。PyTorch 与 Torchvision 需按设备配置对应的 CUDA 版本。

运行命令前，从项目根目录准备正式数据目录 `fittings_dataset/`。YOLO 与 RT-DETR 读取 `data.yaml`，Torchvision 读取 `annotations/instances_*.json`。调整 `data.yaml` 的数据根路径至本机目录。获准访问原始 XML、图像与固定划分时，可用 `scripts/prepare_fittings.py --help` 查看转换方式。论文正式划分为训练 1092 张、验证 234 张、测试 234 张。

COCO 初始化权重的下载地址和校验值见 `models/pretrained/pretrained_manifest.json`。训练后权重按课题组许可提供，取得 `trained-weights.zip` 后解压到仓库根目录，检查 `weights/` 下文件名与清单一致。

## 复现训练与测试

训练以验证集选取检查点。下列命令中的 `--output` 指向新目录。评估时 `--split val` 与 `--split test` 分别对应验证和正式测试，按需指定新输出目录。

### E1 与 E2

Faster R-CNN 训练及固定权重测试：

```bash
python scripts/run_torchvision_baseline.py --data fittings_dataset --family fasterrcnn \
  --weights models/pretrained/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth \
  --output runs/E1_train --imgsz 640 --epochs 60 --batch 2 --workers 0
python scripts/run_torchvision_baseline.py --data fittings_dataset --family fasterrcnn \
  --weights models/pretrained/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth \
  --checkpoint weights/E1_fasterrcnn_640_best.pth --split test \
  --output runs/E1_test --imgsz 640 --workers 0
```

RetinaNet 对应 `--family retinanet`，使用 `models/pretrained/retinanet_resnet50_fpn_coco-eeacb38b.pth` 初始化，固定权重测试时使用 `weights/E2_retinanet_640_best.pth`。

### E3 至 E5

YOLOv8s 训练及固定权重测试：

```bash
python scripts/run_val_baseline.py --data fittings_dataset/data.yaml --family yolo \
  --weights models/pretrained/yolov8s.pt --output runs/E3_train \
  --imgsz 640 --epochs 60 --batch 2 --workers 0
python scripts/run_val_baseline.py --data fittings_dataset/data.yaml --family yolo \
  --weights weights/E3_yolov8s_640_best.pt --eval-only --split test \
  --output runs/E3_test --imgsz 640 --workers 0
```

| 实验 | `--family` | 初始化权重文件名 | 测试检查点文件名 |
| --- | --- | --- | --- |
| E4 | `yolo` | `yolo11s.pt` | `E4_yolo11s_640_best.pt` |
| E5 | `rtdetr` | `rtdetr-l.pt` | `E5_rtdetr_l_640_best.pt` |

### E7 与 E8

```bash
python scripts/run_sliced_baseline.py --data fittings_dataset \
  --weights weights/E3_yolov8s_640_best.pt --split test \
  --tile-size 1024 --overlap 0.20 --output runs/E7_test
python scripts_e8/run_sliced_baseline.py --data fittings_dataset \
  --weights weights/E3_yolov8s_640_best.pt --split test --imgsz 1280 \
  --whole-only --output runs/E8_test
```

E7 同时输出整图 640 与整图加切片，E8 输出整图 1280。两者的耗时来自同一设备的不同运行批次。验证结果见 `results/inference_comparison_val.csv`，测试结果见 `results/inference_comparison_test.csv`；指标定义及来源见 `results/README.md`。
