# 输电线路金具检测

本仓库提供五模型对比和固定 YOLOv8s 权重推理对照的课程论文复现材料。数据所包含的原始图像、标注和划分清单未公开。

- `scripts/` 包含数据整理、训练、预测和共同 COCO 评价入口。
- `configs/` 保存 E1–E5 的实际训练参数，文件路径已改为相对项目根目录的写法。E4 配置保留实际断点续训记录。
- `results/` 提供汇总指标。CSV 中 AP 和 AR 用 0–1 表示。
- `weights-manifest.json` 列出各实验训练后权重的文件名、大小和 SHA-256。E7 与 E8 共用 E3 权重。

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
  --weights weights/E3_yolov8s_640_best.pt --split test --imgsz 640 \
  --tile-size 1024 --overlap 0.20 --output runs/E7_test
python scripts/run_sliced_baseline.py --data fittings_dataset \
  --weights weights/E3_yolov8s_640_best.pt --split test --imgsz 1280 \
  --whole-only --output runs/E8_test
```

E7 同时输出整图 640 与整图加切片，E8 输出整图 1280。两者的耗时来自同一设备的不同运行批次。汇总结果见 `results/inference_comparison.csv`。
