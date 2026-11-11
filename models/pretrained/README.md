# 官方预训练权重

此目录提供下载说明与校验程序，不包含 .pt/.pth 文件。下载后将下列文件放在本目录，保留原文件名。它们用于开始训练，项目训练后的权重在 Release 中。

| 实验 | 下载文件 |
| --- | --- |
| E1 | [fasterrcnn_resnet50_fpn_coco-258fb6c6.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) |
| E2 | [retinanet_resnet50_fpn_coco-eeacb38b.pth](https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth) |
| E3 | [yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt) |
| E4 | [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| E5 | [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt) |

下载完成后，在项目根目录运行：

```bash
python models/pretrained/verify_weights.py
```

该命令只校验文件，不执行下载。清单中的 SHA-256 是准备实验材料时记录的文件指纹。
