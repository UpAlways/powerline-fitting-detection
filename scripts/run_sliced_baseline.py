"""Compare whole-image and whole-plus-tile inference using one frozen YOLO checkpoint."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluate_fittings import evaluate
from run_val_baseline import digest, write_json


CONF = 0.001
NMS_IOU = 0.5
MAX_DET = 300


def tile_starts(length, size, step):
    last = max(0, length - size)
    return [*range(0, last, step), last]


def tile_windows(width, height, size, overlap):
    step = max(1, round(size * (1 - overlap)))
    windows = [
        (x, y, min(x + size, width), min(y + size, height))
        for y in tile_starts(height, size, step)
        for x in tile_starts(width, size, step)
    ]
    return [window for window in windows if window != (0, 0, width, height)]


def shift_boxes(detections, window):
    x, y, right, bottom = window
    shifted = detections.copy()
    shifted[:, (0, 2)] = np.clip(shifted[:, (0, 2)], 0, right - x) + x
    shifted[:, (1, 3)] = np.clip(shifted[:, (1, 3)], 0, bottom - y) + y
    return shifted[(shifted[:, 2] > shifted[:, 0]) & (shifted[:, 3] > shifted[:, 1])]


def coco_rows(detections, entry):
    return [
        {
            "image_id": entry["id"],
            "file_name": entry["file_name"],
            "category_id": int(cls) + 1,
            "bbox": [round(float(v), 3) for v in (x1, y1, x2 - x1, y2 - y1)],
            "score": round(float(score), 5),
        }
        for x1, y1, x2, y2, score, cls in detections
    ]


def small_target_diagnostic(images, annotations, predictions):
    """Match the 640-scale, short-side <16 px diagnostic used for selecting E7."""
    scales = {entry["id"]: 640 / max(entry["height"], entry["width"]) for entry in images}
    targets = defaultdict(list)
    small = set()
    for target in annotations:
        image_id = target["image_id"]
        if image_id not in scales or target.get("iscrowd"):
            continue
        targets[image_id, target["category_id"]].append(target)
        if min(target["bbox"][2:]) * scales[image_id] < 16:
            small.add(target["id"])
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred["image_id"]].append(pred)
    matched = set()
    for image_id, detections in by_image.items():
        for pred in sorted(detections, key=lambda p: -p["score"])[:100]:
            x, y, w, h = pred["bbox"]
            best_id, best_iou = None, 0.5
            for target in targets[image_id, pred["category_id"]]:
                if target["id"] in matched:
                    continue
                u, v, a, b = target["bbox"]
                intersection = max(0, min(x + w, u + a) - max(x, u)) * max(0, min(y + h, v + b) - max(y, v))
                iou = intersection / (w * h + a * b - intersection)
                if iou >= best_iou:
                    best_id, best_iou = target["id"], iou
            if best_id is not None:
                matched.add(best_id)
    hits = len(small & matched)
    return {"targets": len(small), "matched": hits, "hit_rate": hits / len(small) if small else None}


def save_preview(path, image, ground_truth, predictions):
    import cv2
    from ultralytics.utils.patches import imwrite

    height, width = image.shape[:2]
    scale = min(1, 1280 / max(width, height))
    canvas = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    for rows, color in ((ground_truth, (0, 190, 0)), (predictions, (0, 0, 255))):
        for row in rows:
            if row.get("score", 1) < 0.25:
                continue
            x, y, w, h = row["bbox"]
            cv2.rectangle(canvas, (round(x * scale), round(y * scale)),
                          (round((x + w) * scale), round((y + h) * scale)), color, 2)
    if not imwrite(str(path), canvas):
        raise OSError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="formal dataset directory")
    parser.add_argument("--weights", type=Path, required=True, help="E3 weights/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--limit", type=int, help="validation smoke only, skips COCO scores")
    parser.add_argument("--preview", type=int, default=0, help="number of original images to preview in both modes")
    args = parser.parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < 1 or args.preview < 0:
        parser.error("provide positive tile size, overlap in [0,1), and a nonnegative preview count")
    if args.limit is not None and (args.split != "val" or args.limit <= 0):
        parser.error("--limit is only for a positive validation smoke subset")
    if args.output.exists() or not args.weights.is_file() or args.weights.suffix != ".pt":
        parser.error("provide a new output directory and an existing .pt checkpoint")

    data = args.data.resolve(strict=True)
    selection_path = data / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("purpose") != "formal" or selection.get("layout") != "train-val-test":
        parser.error("use the formal three-way split")
    annotation_path = data / "annotations" / f"instances_{args.split}.json"
    truth = json.loads(annotation_path.read_text(encoding="utf-8"))
    entries = sorted(truth["images"], key=lambda item: item["id"])
    selected = selection["images"][args.split]
    filenames = [Path(entry["file_name"]).name for entry in entries]
    if not entries or len({entry["id"] for entry in entries}) != len(entries) or sorted(filenames) != sorted(selected):
        parser.error("COCO images do not match the frozen split")
    entries = entries[:args.limit]

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import torch
    import ultralytics
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    from ultralytics.utils.patches import imread

    if ultralytics.__version__ != "8.3.221":
        parser.error(f"expected Ultralytics 8.3.221, found {ultralytics.__version__}")
    if args.device == "0" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.limit is None:
        import pycocotools.coco  # fail before inference if scoring is unavailable

    model = YOLO(str(args.weights.resolve()))
    expected = {row["id"] - 1: row["name"] for row in truth["categories"]}
    if len(expected) != 18 or model.task != "detect" or dict(model.names) != expected:
        parser.error("checkpoint does not match the 18-class detection dataset")

    def predict(view):
        result = model.predict(view, imgsz=640, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                               rect=False, batch=1, half=False, agnostic_nms=False,
                               device=args.device, save=False, verbose=False)[0]
        return result.boxes.data.detach().cpu().numpy()

    def clock():
        if args.device == "0":
            torch.cuda.synchronize()
        return perf_counter()

    args.output.mkdir(parents=True)
    modes = ("whole", "sliced")
    predictions = {mode: [] for mode in modes}
    timings = {mode: [] for mode in modes}
    for mode in modes:
        (args.output / mode).mkdir()
    if args.preview:
        (args.output / "previews").mkdir()
    view_counts = []
    ground_truth = defaultdict(list)
    for annotation in truth["annotations"]:
        ground_truth[annotation["image_id"]].append(annotation)

    for index, entry in enumerate(entries, 1):
        relative = Path(entry["file_name"])
        path = (data / relative).resolve(strict=True)
        if relative.parts[:2] != ("images", args.split) or data not in path.parents:
            raise ValueError(f"Invalid image path {relative}")
        image = imread(str(path))
        width, height = entry["width"], entry["height"]
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"Cannot decode image at annotated size: {path}")
        if index == 1:
            for _ in range(3):
                predict(image)
        start = clock()
        full = predict(image)
        whole_done = clock()
        windows = tile_windows(width, height, args.tile_size, args.overlap)
        fused = full
        if windows:
            candidates = [full]
            for window in windows:
                x, y, right, bottom = window
                candidates.append(shift_boxes(predict(image[y:bottom, x:right]), window))
            rows = torch.from_numpy(np.concatenate(candidates))
            keep = batched_nms(rows[:, :4], rows[:, 4], rows[:, 5].long(), NMS_IOU)
            fused = rows[keep[:MAX_DET]].numpy()
        sliced_done = clock()
        timings["whole"].append((whole_done - start) * 1000)
        timings["sliced"].append((sliced_done - start) * 1000)
        view_counts.append(1 + len(windows))
        for mode, boxes in zip(modes, (full, fused)):
            rows = coco_rows(boxes, entry)
            predictions[mode].extend(rows)
            if index <= args.preview:
                save_preview(args.output / "previews" / f"{entry['id']:04d}_{mode}.jpg",
                             image, ground_truth[entry["id"]], rows)
        print(f"{index}/{len(entries)} {relative.name}: {view_counts[-1]} views, "
              f"{len(full)} whole boxes, {len(fused)} fused boxes", flush=True)

    metadata = {
        "split": args.split, "image_count": len(entries), "input_size": 640,
        "checkpoint": str(args.weights.resolve()), "checkpoint_sha256": digest(args.weights),
        "selection_sha256": digest(selection_path), "annotation_sha256": digest(annotation_path),
        "device": args.device, "torch": str(torch.__version__), "ultralytics": ultralytics.__version__,
        "conf": CONF, "nms_iou": NMS_IOU, "max_det": MAX_DET, "tile_size": args.tile_size,
        "tile_overlap": args.overlap, "tile_step": max(1, round(args.tile_size * (1 - args.overlap))),
        "tile_border": "shift final tile to image edge", "merge": "class-aware NMS", "rect": False,
        "timing_scope": "decoded image to original-coordinate boxes; includes slicing and fusion",
        "warmup_views": 3, "smoke_only": args.limit is not None,
    }
    comparison = {"config": metadata, "results": {}}
    for mode in modes:
        rows = predictions[mode]
        directory = args.output / mode
        write_json(directory / f"predictions_{args.split}.json", rows)
        summary = {
            "ms_per_original_image": float(np.mean(timings[mode])),
            "views_per_original_image": float(np.mean(view_counts)) if mode == "sliced" else 1.0,
            "small_target_diagnostic": small_target_diagnostic(entries, truth["annotations"], rows),
        }
        if args.limit is None:
            scores = evaluate(annotation_path, rows)
            write_json(directory / f"metrics_{args.split}.json", {**scores, **metadata, **summary, "mode": mode})
            summary.update({key: scores[key] for key in ("mAP50", "mAP50_95", "AR100")})
        comparison["results"][mode] = summary
    output = args.output / "comparison.json"
    write_json(output, comparison)
    print(f"{'SMOKE' if args.limit is not None else args.split.upper()} OK: {output}", flush=True)


if __name__ == "__main__":
    main()
