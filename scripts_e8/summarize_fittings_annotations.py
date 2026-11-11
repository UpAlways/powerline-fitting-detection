"""Summarize a COCO split without exporting image names or individual boxes."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def summarize(data):
    images = {row["id"]: (row["width"], row["height"]) for row in data["images"]}
    categories = {row["id"]: row["name"] for row in data["categories"]}
    if len(images) != len(data["images"]) or len(categories) != len(data["categories"]):
        raise ValueError("duplicate image or category ID")
    counts = Counter()
    small_short = Counter()
    areas = Counter()
    invalid = 0
    for row in data["annotations"]:
        box = row["bbox"]
        size = images.get(row["image_id"])
        category = row["category_id"]
        if size is None or category not in categories or len(box) != 4 or not all(math.isfinite(v) for v in box):
            invalid += 1
            continue
        x, y, width, height = box
        if (min(size) <= 0 or min(width, height) <= 0 or min(x, y) < 0
                or x + width > size[0] + 1e-6 or y + height > size[1] + 1e-6):
            invalid += 1
            continue
        counts[category] += 1
        area = width * height
        areas["lt_32_squared" if area < 32**2 else "lt_96_squared" if area < 96**2 else "ge_96_squared"] += 1
        if min(width, height) * 640 / max(size) < 16:
            small_short[category] += 1
    return {
        "images": len(images), "classes": len(categories), "annotations": len(data["annotations"]),
        "invalid_annotations": invalid,
        "bbox_area_original_px2": {key: areas[key] for key in ("lt_32_squared", "lt_96_squared", "ge_96_squared")},
        "short_side_lt_16_px_at_long_side_640": sum(small_short.values()),
        "per_class": [{"category_id": category, "name": name, "instances": counts[category],
                       "short_side_lt_16_px_at_long_side_640": small_short[category]}
                      for category, name in sorted(categories.items(), key=lambda item: (counts[item[0]], item[0]))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.annotations.read_bytes()
    result = summarize(json.loads(raw))
    result["annotation_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"{result['images']} images, {result['classes']} classes, {result['annotations']} annotations, "
          f"{result['invalid_annotations']} invalid; wrote {args.output}")


if __name__ == "__main__":
    main()
