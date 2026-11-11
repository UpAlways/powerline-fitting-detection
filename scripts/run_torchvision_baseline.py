"""Fine-tune or evaluate Torchvision detection models on the exported COCO splits."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms.functional import to_tensor

from evaluate_fittings import evaluate


class FittingsDataset(Dataset):
    def __init__(self, root, split, label_offset):
        self.annotation = root / "annotations" / ("instances_" + split + ".json")
        truth = json.loads(self.annotation.read_text(encoding="utf-8"))
        self.categories = {row["id"]: row["name"] for row in truth["categories"]}
        self.images = [(row["id"], root / row["file_name"]) for row in truth["images"]]
        self.boxes = {image_id: [] for image_id, _ in self.images}
        self.labels = {image_id: [] for image_id, _ in self.images}
        for row in truth["annotations"]:
            x, y, width, height = row["bbox"]
            self.boxes[row["image_id"]].append([x, y, x + width, y + height])
            self.labels[row["image_id"]].append(row["category_id"] - label_offset)
        if set(self.categories) != set(range(1, 19)) or not self.images:
            raise ValueError("expected 18 consecutive COCO category IDs and a nonempty split")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_id, path = self.images[index]
        with Image.open(path) as image:
            pixels = to_tensor(image.convert("RGB"))
        return pixels, {
            "image_id": image_id,
            "boxes": torch.tensor(self.boxes[image_id], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(self.labels[image_id], dtype=torch.int64),
        }


def collate(batch):
    return tuple(zip(*batch))


def build_model(family, weights_path, size, class_count):
    faster = family == "fasterrcnn"
    if faster:
        factory = detection.fasterrcnn_resnet50_fpn
        weights = detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        options = {"box_score_thresh": 0.001, "box_nms_thresh": 0.5, "box_detections_per_img": 300}
    else:
        factory = detection.retinanet_resnet50_fpn
        weights = detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1
        options = {"score_thresh": 0.001, "nms_thresh": 0.5, "detections_per_img": 300}

    # Use the factory's COCO configuration, including its normalization layers,
    # while supplying the already uploaded state dict in place of a download.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    with patch.object(type(weights), "get_state_dict", return_value=state):
        model = factory(weights=weights, min_size=size, max_size=size, **options)

    if faster:
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, class_count + 1)
    else:
        old_head = model.head.classification_head
        head = RetinaNetClassificationHead(
            model.backbone.out_channels, old_head.num_anchors, class_count
        )
        head.conv = old_head.conv
        model.head.classification_head = head
    return model


@torch.inference_mode()
def predict(model, loader, device, label_offset):
    model.eval()
    predictions = []
    for images, targets in loader:
        detections = model([image.to(device) for image in images])
        for result, target in zip(detections, targets):
            boxes = torchvision.ops.box_convert(result["boxes"], "xyxy", "xywh").cpu().tolist()
            labels = result["labels"].cpu().tolist()
            scores = result["scores"].cpu().tolist()
            predictions.extend(
                {"image_id": target["image_id"], "category_id": label + label_offset,
                 "bbox": box, "score": score}
                for box, label, score in zip(boxes, labels, scores)
            )
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset root containing annotations/")
    parser.add_argument("--weights", type=Path, required=True, help="local official COCO weights")
    parser.add_argument("--family", choices=("fasterrcnn", "retinanet"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="evaluate this trained checkpoint without training")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.split == "test" and args.checkpoint is None:
        parser.error("test requires an already trained checkpoint")
    if args.output.exists():
        parser.error("output already exists; choose a new run directory")
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error("epochs, image size and batch must be positive and workers nonnegative")
    if not args.weights.is_file() or (args.checkpoint is not None and not args.checkpoint.is_file()):
        parser.error("provide existing local weights and checkpoint files")

    import pycocotools.cocoeval  # check before training

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    torch.manual_seed(42)
    label_offset = int(args.family == "retinanet")
    evaluation = FittingsDataset(args.data, args.split, label_offset)
    eval_loader = DataLoader(evaluation, batch_size=1, collate_fn=collate, num_workers=args.workers)
    selection = args.data / "selection.json"
    split_hash = hashlib.sha256(selection.read_bytes()).hexdigest()
    model = build_model(args.family, args.weights, args.imgsz, len(evaluation.categories))
    identity = {"family": args.family, "input_size": args.imgsz, "selection_sha256": split_hash,
                "categories": evaluation.categories}

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            parser.error("checkpoint model, size, categories or dataset selection does not match")
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        args.output.mkdir(parents=True)
        predictions = predict(model, eval_loader, device, label_offset)
        scores = evaluate(evaluation.annotation, predictions)
        save_results(args.output, args.split, predictions, scores)
        return

    training = FittingsDataset(args.data, "train", label_offset)
    if training.categories != evaluation.categories:
        parser.error("training and validation category maps differ")
    loader = DataLoader(training, batch_size=args.batch, shuffle=True,
                        collate_fn=collate, num_workers=args.workers)
    model.to(device)
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    accumulation = max(1, round(8 / args.batch))
    weights_dir = args.output / "weights"
    weights_dir.mkdir(parents=True)
    config = {**vars(args), "optimizer": "SGD", "initial_lr": 0.005, "seed": 42,
              "accumulation_steps": accumulation,
              "torch": torch.__version__, "torchvision": torchvision.__version__}
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(loader, 1):
            targets = [{k: t[k].to(device) for k in ("boxes", "labels")} for t in targets]
            loss = sum(model([image.to(device) for image in images], targets).values())
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}, step {step}")
            window_start = (step - 1) // accumulation * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            if step == window_start + window_size:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += value
            if step == 1 or step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss={value:.4f}", flush=True)
        scheduler.step()
        predictions = predict(model, eval_loader, device, label_offset)
        scores = evaluate(evaluation.annotation, predictions)
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "loss": total_loss / len(loader), **scores}) + "\n")
        checkpoint = {"identity": identity, "model": model.state_dict(), "epoch": epoch}
        save_checkpoint(weights_dir / "last.pth", checkpoint)
        if scores["mAP50_95"] > best:
            best = scores["mAP50_95"]
            save_checkpoint(weights_dir / "best.pth", checkpoint)
            save_results(args.output, "val", predictions, scores)
        print(f"Epoch {epoch}/{args.epochs} validation mAP50-95={scores['mAP50_95']:.4f}", flush=True)


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_results(output, split, predictions, scores):
    for stem, result in (("predictions", predictions), ("metrics", scores)):
        path = output / f"{stem}_{split}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{split.upper()} OK: {output / ('metrics_' + split + '.json')}", flush=True)


if __name__ == "__main__":
    main()
