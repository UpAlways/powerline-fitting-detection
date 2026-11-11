"""Score predictions with the common COCO bbox evaluator."""

import argparse
import copy
import json
import math
from pathlib import Path


def evaluate(annotation_path, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO(str(annotation_path))
    image_ids = set(truth.imgs)
    category_ids = set(truth.cats)
    if not image_ids or not category_ids:
        raise ValueError("empty ground truth image or category list")
    for row in predictions:
        if row["image_id"] not in image_ids or row["category_id"] not in category_ids:
            raise ValueError("prediction refers to an unknown image or class")
        box, score = row["bbox"], row["score"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) < 0:
            raise ValueError("prediction bbox must be finite xywh with nonnegative size")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("prediction score outside [0,1]")
    if predictions:
        detected = truth.loadRes(copy.deepcopy(predictions))
    else:
        detected = COCO()
        detected.dataset = copy.deepcopy(truth.dataset)
        detected.dataset["annotations"] = []
        detected.createIndex()
    evaluator = COCOeval(truth, detected, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = sorted(category_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    per_class = {}
    for i, category in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, i, 0, -1]
        values = precision[precision >= 0]
        values50 = precision[0][precision[0] >= 0]
        per_class[truth.cats[category]["name"]] = {
            "AP50_95": float(values.mean()) if values.size else None,
            "AP50": float(values50.mean()) if values50.size else None,
        }
    return {"mAP50_95": float(evaluator.stats[0]), "mAP50": float(evaluator.stats[1]),
            "AP_small": float(evaluator.stats[3]), "AP_medium": float(evaluator.stats[4]),
            "AP_large": float(evaluator.stats[5]),
            "AR100": float(evaluator.stats[8]), "per_class": per_class,
            "image_count": len(image_ids), "prediction_count": len(predictions),
            "maxDets": list(evaluator.params.maxDets), "unit": "fraction_0_to_1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    scores = evaluate(args.annotations, json.loads(args.predictions.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
