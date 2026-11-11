"""Train with validation or score a frozen Ultralytics detector on a chosen split."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from evaluate_fittings import evaluate


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def coco_id_validator(base_validator, images_by_path):
    """Keep the original filenames and assign COCO IDs independently."""
    class Validator(base_validator):
        def pred_to_json(self, predn, pbatch):
            path = Path(pbatch["im_file"]).resolve()
            if path not in images_by_path:
                raise ValueError("evaluation image not in the COCO split: {}".format(path))
            image_id, relative_path = images_by_path[path]
            start = len(self.jdict)
            super().pred_to_json(predn, pbatch)
            for row in self.jdict[start:]:
                row["image_id"] = image_id
                row["file_name"] = relative_path

    return Validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--family", choices=("yolo", "rtdetr"), default="yolo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-only", action="store_true", help="evaluate an existing 18-class checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.split == "test" and not args.eval_only:
        parser.error("test is reserved for --eval-only with a frozen checkpoint")
    data = args.data.resolve(strict=True)
    weights = args.weights.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new directory")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs and batch must be positive and workers nonnegative")
    if not data.is_file() or not weights.is_file() or weights.suffix != ".pt":
        parser.error("provide a local data.yaml and a local .pt file")
    selection_path = data.parent / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    layout = selection.get("layout")
    if selection.get("purpose") != "formal" or layout not in ("train-val", "train-val-test"):
        parser.error("use a full formal export, not a draft or a smoke set")
    expected_splits = {"train", "val", "test"} if layout == "train-val-test" else {"train", "val"}
    if set(selection.get("images", {})) != expected_splits:
        parser.error("split list does not match the exported layout")
    if args.split not in expected_splits:
        parser.error("the requested split is not in this data export")
    annotation_path = data.parent / "annotations" / ("instances_" + args.split + ".json")
    if not annotation_path.is_file():
        # Older exports used val.json/test.json; read them without rewriting.
        annotation_path = data.parent / "annotations" / (args.split + ".json")
    truth = json.loads(annotation_path.read_text(encoding="utf-8"))
    expected_names = {entry["id"] - 1: entry["name"] for entry in truth["categories"]}
    if len(expected_names) != 18 or not truth["images"]:
        parser.error("expected 18 categories and a non-empty evaluation split")
    if len(truth["images"]) != len(selection["images"][args.split]):
        parser.error("annotation count does not match the frozen image list")
    images_by_path = {}
    for entry in truth["images"]:
        path = (data.parent / entry["file_name"]).resolve(strict=True)
        if data.parent not in path.parents or path in images_by_path or not path.is_file():
            parser.error("invalid or repeated COCO image path: {}".format(path))
        images_by_path[path] = (entry["id"], entry["file_name"])
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    import pycocotools.coco  # fail before a long training run if evaluation is unavailable
    import torch
    import ultralytics
    from ultralytics import RTDETR, YOLO
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.models.yolo.detect.val import DetectionValidator

    if ultralytics.__version__ != "8.3.221":
        parser.error("expected Ultralytics 8.3.221; installed {}".format(ultralytics.__version__))
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the active environment")
    rtdetr = args.family == "rtdetr"
    model_type = RTDETR if rtdetr else YOLO

    class RTDETRJsonValidator(RTDETRValidator):
        def scale_preds(self, predn, pbatch):
            # In 8.3.221 RT-DETR's JSON exporter already rescales boxes from
            # the stretched square input to original width and height.
            return predn

    validator_type = coco_id_validator(RTDETRJsonValidator if rtdetr else DetectionValidator, images_by_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Plots are off; skip the framework's optional font lookup/download.
    with patch("ultralytics.data.utils.check_font", return_value=None):
        if args.eval_only:
            output.mkdir()
            checkpoint = weights
        else:
            model = model_type(str(weights))
            model.train(data=str(data), split="val", epochs=args.epochs, patience=args.epochs,
                        imgsz=args.imgsz, batch=args.batch, nbs=8, device=args.device,
                        workers=args.workers, seed=42, amp=False, plots=False, cache=False,
                        deterministic=not rtdetr, optimizer="AdamW" if rtdetr else "SGD",
                        lr0=0.0001 if rtdetr else 0.01,
                        weight_decay=0.0001 if rtdetr else 0.0005,
                        conf=0.001, iou=0.5, max_det=300, save=True, val=True,
                        project=str(output.parent), name=output.name, exist_ok=False)
            output = Path(model.trainer.save_dir)
            checkpoint = output / "weights" / "best.pt"
            if not checkpoint.is_file():
                raise RuntimeError("best.pt not found: {}".format(checkpoint))
        best = model_type(str(checkpoint))
        if dict(best.names) != expected_names:
            raise ValueError("checkpoint categories do not match the evaluation annotations")
        saved = {}

        def capture(**kwargs):
            validator = validator_type(**kwargs)
            saved["validator"] = validator
            return validator

        metric = best.val(validator=capture, data=str(data), split=args.split, imgsz=args.imgsz,
                          batch=1, device=args.device, workers=args.workers, half=False,
                          conf=0.001, iou=0.5, max_det=300, save_json=True, plots=False,
                          project=str(output), name="evaluation_" + args.split, exist_ok=False)
    validator = saved["validator"]
    if validator.seen != len(truth["images"]):
        raise RuntimeError("evaluation did not process every image: {} vs {}".format(validator.seen, len(truth["images"])))
    predictions = validator.jdict
    write_json(output / ("predictions_" + args.split + ".json"), predictions)
    scores = evaluate(annotation_path, predictions)
    scores.update({"split": args.split, "layout": layout, "has_reserved_test_split": "test" in expected_splits,
                   "checkpoint_rule": "provided checkpoint" if args.eval_only else "Ultralytics validation mAP50-95",
                   "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                   "selection_sha256": digest(selection_path),
                   "annotation_sha256": digest(annotation_path), "torch": torch.__version__,
                   "ultralytics": ultralytics.__version__, "input_size": args.imgsz,
                   "ultralytics_metrics": metric.results_dict})
    metric_file = output / ("metrics_" + args.split + ".json")
    write_json(metric_file, scores)
    print("{} OK: {}".format(args.split.upper(), metric_file), flush=True)
    print("mAP50={:.4f}  mAP50-95={:.4f}  AR100={:.4f}".format(scores["mAP50"], scores["mAP50_95"], scores["AR100"]), flush=True)


if __name__ == "__main__":
    main()
