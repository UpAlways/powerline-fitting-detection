"""Audit VOC-style XML, draft image splits, or export a supplied formal split."""

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

try:
    from PIL import Image
except ImportError:
    Image = None


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def index_files(files, key):
    result = defaultdict(list)
    for path in files:
        result[key(path)].append(path)
    return result


def relkey(path, root):
    return path.relative_to(root).with_suffix("").as_posix().casefold()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def exact_copy_aliases(images, xmls, image_root, xml_root):
    """Remove a (N) alias from the working list only if both files match."""
    image_keys = index_files(images, lambda p: relkey(p, image_root))
    xml_keys = index_files(xmls, lambda p: relkey(p, xml_root))
    skip_images, skip_xmls, matches = set(), set(), []
    for image in images:
        match = re.fullmatch(r"(.+)\([0-9]+\)", image.stem)
        if not match:
            continue
        alias_key = relkey(image, image_root)
        original_key = relkey(image.with_name(match[1] + image.suffix), image_root)
        choices = [image_keys[alias_key], image_keys.get(original_key, []),
                   xml_keys.get(alias_key, []), xml_keys.get(original_key, [])]
        if any(len(items) != 1 for items in choices):
            continue
        _, original, xml, original_xml = (items[0] for items in choices)
        if image.stat().st_size != original.stat().st_size or xml.read_bytes() != original_xml.read_bytes():
            continue
        if sha256_file(image) != sha256_file(original):
            continue
        skip_images.add(image)
        skip_xmls.add(xml)
        matches.append({"image": image.relative_to(image_root).as_posix(),
                        "xml": xml.relative_to(xml_root).as_posix(),
                        "kept_image": original.relative_to(image_root).as_posix()})
    return [p for p in images if p not in skip_images], [p for p in xmls if p not in skip_xmls], matches


def dir_splits(path, root):
    names = [root.name] + list(path.relative_to(root).parts[:-1])
    return {SPLITS[name.casefold()] for name in names if name.casefold() in SPLITS}


def split_lists(folder, images, image_root):
    if folder is None:
        candidate = image_root.parent / "ImageSets" / "Main"
        if not (candidate / "train.txt").is_file():
            return {}, False
        folder = candidate
    if not folder.is_dir():
        raise ValueError("split directory does not exist: {}".format(folder))
    by_stem = index_files(images, lambda p: p.stem.casefold())
    by_rel = index_files(images, lambda p: relkey(p, image_root))
    assigned = {}
    for split in ("train", "val", "test"):
        source = folder / (split + ".txt")
        if not source.is_file():
            continue
        for raw in source.read_text(encoding="utf-8-sig").splitlines():
            key = raw.strip().replace("\\", "/").casefold()
            if not key:
                continue
            p = PurePosixPath(key)
            if p.suffix in EXTS:
                p = p.with_suffix("")
            matches = by_rel.get(p.as_posix(), []) or by_stem.get(p.name, [])
            if len(matches) != 1:
                raise ValueError("unknown or ambiguous image in {}: {}".format(source.name, raw))
            if matches[0] in assigned and assigned[matches[0]] != split:
                raise ValueError("image appears in multiple splits: {}".format(raw))
            assigned[matches[0]] = split
    if not assigned:
        raise ValueError("no split entries in {}".format(folder))
    return assigned, True


def audit(image_root, xml_root, coords, lists):
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    xmls = sorted(p for p in xml_root.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")
    if not images or not xmls:
        raise ValueError("found {} images and {} XML files; check both paths".format(len(images), len(xmls)))
    image_count, xml_count = len(images), len(xmls)
    images, xmls, alias_copies = exact_copy_aliases(images, xmls, image_root, xml_root)
    assigned, explicit = split_lists(lists, images, image_root)
    tagged = explicit or any(dir_splits(p, image_root) for p in images) or any(dir_splits(p, xml_root) for p in xmls)
    by_rel = index_files(images, lambda p: relkey(p, image_root))
    by_name = index_files(images, lambda p: p.name.casefold())
    by_stem = index_files(images, lambda p: p.stem.casefold())
    issues, samples, classes, raw_sizes = Counter(), defaultdict(list), Counter(), Counter()
    coordinates, split_counts = Counter(), Counter()
    pairs, paired, repeated = {}, set(), set()
    first_xml = {}
    duplicate_count, duplicate_examples = 0, []

    def note(kind, name):
        issues[kind] += 1
        if len(samples[kind]) < 4:
            samples[kind].append(name)

    for xml in xmls:
        name = xml.relative_to(xml_root).as_posix()
        try:
            raw = xml.read_bytes()
            root = ET.fromstring(raw)
        except (ET.ParseError, OSError) as error:
            note("unreadable_xml", name + ": " + str(error))
            continue
        if root.tag != "annotation":
            note("unsupported_xml_root", name + ": " + root.tag)
            continue
        filename = (root.findtext("filename") or "").strip().replace("\\", "/").split("/")[-1].casefold()
        declared = by_name.get(filename, [])
        relative = by_rel.get(relkey(xml, xml_root), [])
        matches = relative or declared or by_stem.get(xml.stem.casefold(), [])
        if len(matches) != 1 or (declared and matches[0] not in declared):
            note("missing_or_ambiguous_image", name)
            continue
        image = matches[0]
        paired.add(image)
        tags = dir_splits(image, image_root) | dir_splits(xml, xml_root)
        if explicit:
            split = assigned.get(image)
            if split and tags and tags != {split}:
                repeated.add(image)
                note("conflicting_split", name)
                continue
        else:
            if len(tags) > 1:
                repeated.add(image)
                note("conflicting_split", name)
                continue
            split = next(iter(tags), None)
        digest = hashlib.sha256(raw).digest()
        if image in first_xml:
            first_name, first_digest, first_split = first_xml[image]
            if split != first_split:
                repeated.add(image)
                note("conflicting_split", first_name + " <> " + name)
            elif digest == first_digest:
                duplicate_count += 1
                if len(duplicate_examples) < 4:
                    duplicate_examples.append([first_name, name])
            else:
                repeated.add(image)
                note("conflicting_xml_for_image", first_name + " <> " + name)
            continue
        first_xml[image] = (name, digest, split)
        try:
            xml_w, xml_h = int(root.findtext("size/width")), int(root.findtext("size/height"))
            if xml_w <= 0 or xml_h <= 0:
                raise ValueError("nonpositive dimensions")
        except (TypeError, ValueError):
            note("missing_xml_size", name)
            continue
        w, h, orientation = xml_w, xml_h, 1
        if Image is not None:
            try:
                with Image.open(image) as im:
                    w, h = im.size
                    orientation = im.getexif().get(274, 1)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                note("unreadable_image", image.name + ": " + str(error))
                continue
        raw_sizes["{}x{}".format(w, h)] += 1
        if (w, h) != (xml_w, xml_h):
            note("size_mismatch", name + ": image={}x{}, XML={}x{}".format(w, h, xml_w, xml_h))
            continue
        if orientation not in (1, None):
            note("exif_orientation_needs_review", image.name)
            continue
        objects = root.findall("object")
        boxes, invalid = [], False
        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            try:
                x1, y1, x2, y2 = [float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                if not label or not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    raise ValueError("missing label or non-finite box")
            except (ValueError, TypeError):
                note("unsupported_or_invalid_box", name)
                invalid = True
                continue
            coordinates["boxes_with_zero_min"] += int(x1 == 0 or y1 == 0)
            coordinates["boxes_with_max_at_image_edge"] += int(x2 == w or y2 == h)
            coordinates["difficult_objects"] += int(obj.findtext("difficult") == "1")
            if coords == "voc1-inclusive":
                x1, y1 = x1 - 1, y1 - 1
            elif coords == "zero-inclusive":
                x2, y2 = x2 + 1, y2 + 1
            # Audit without a coordinate convention allows inclusive single-pixel boxes.
            bad_area = (x2 <= x1 or y2 <= y1) if coords else (x2 < x1 or y2 < y1)
            if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h) or bad_area:
                note("out_of_bounds_or_invalid_box", name)
                invalid = True
                continue
            boxes.append((label, x1, y1, x2 - x1, y2 - y1))
        if not objects:
            note("empty_xml_needs_review", name)
        if boxes and not invalid:
            pairs[image] = (image, w, h, split, boxes)

    records = [r for p, r in pairs.items() if p not in repeated]
    for image, w, h, split, boxes in records:
        split_counts[split or "unassigned"] += 1
        classes.update(b[0] for b in boxes)
    orphans = [p for p in images if p not in paired]
    if orphans:
        issues["images_without_matched_xml"] = len(orphans)
        samples["images_without_matched_xml"] = [p.relative_to(image_root).as_posix() for p in orphans[:4]]
    return records, tagged, classes, {
        "images": image_count, "xml": xml_count, "usable_positive_pairs": len(records),
        "class_count": len(classes), "instances_by_class": dict(sorted(classes.items())),
        "split_counts_usable": dict(split_counts), "detected_split_structure": tagged,
        "common_image_sizes": dict(raw_sizes.most_common(8)), "coordinates": dict(coordinates),
        "image_metadata_checked_with_pillow": Image is not None,
        "identical_xml_duplicates_ignored": duplicate_count,
        "identical_image_xml_pairs_ignored": len(alias_copies),
        "identical_image_xml_pair_examples": alias_copies[:4],
        "duplicate_xml_examples": duplicate_examples,
        "issues": dict(issues), "examples": dict(samples),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def destination(args):
    if args.out is None:
        raise ValueError("--out is required")
    out = args.out.resolve()
    for root in (args.images, args.xml):
        if out == root or root in out.parents:
            raise ValueError("output must be outside both original data folders")
    if out.exists():
        raise ValueError("output already exists; choose a new folder: {}".format(out))
    return out


def draft_split(args, records, tagged, classes, report):
    if tagged:
        raise ValueError("existing split structure detected; do not create a new draft")
    if report["issues"] or len(classes) != args.expected_classes:
        raise ValueError("review the audit issues and class count before drafting a split")
    out = destination(args)
    names = ("train", "val") if args.layout == "train-val" else ("train", "val", "test")
    weights = (0.80, 0.20) if args.layout == "train-val" else (0.70, 0.15, 0.15)
    n_splits = len(names)
    count = len(records)
    targets = [int(count * w) for w in weights]
    targets[0] += count - sum(targets)

    # Hash equal-size candidates only: byte-identical files have equal lengths.
    size_buckets = defaultdict(list)
    for row in records:
        size_buckets[row[0].stat().st_size].append(row)
    clusters = []
    for bucket in size_buckets.values():
        if len(bucket) == 1:
            clusters.append(bucket)
        else:
            same_bytes = defaultdict(list)
            for row in bucket:
                same_bytes[sha256_file(row[0])].append(row)
            clusters.extend(same_bytes.values())
    labels_per_group = [Counter(label for r in group for label in sorted({b[0] for b in r[4]})) for group in clusters]
    group_coverage = Counter(label for labels in labels_per_group for label in labels)
    impossible = sorted(label for label in classes if group_coverage[label] < n_splits)
    if impossible:
        raise ValueError("fewer than {} independent byte groups for: {}".format(n_splits, ", ".join(impossible)))
    label_totals = Counter()
    for labels in labels_per_group:
        label_totals.update(labels)

    best = None
    for attempt in range(16):
        rng = random.Random(42 + attempt)
        order = sorted(range(len(clusters)), key=lambda i: (
            min(group_coverage[c] for c in labels_per_group[i]), -len(clusters[i]), rng.random()))
        rows = [[] for _ in names]
        observed = [Counter() for _ in names]
        for i in order:
            group, labels = clusters[i], labels_per_group[i]

            def score(s):
                overflow = max(0, len(rows[s]) + len(group) - targets[s])
                count_cost = (2 * (len(rows[s]) - targets[s]) + len(group)) * len(group) / max(1, targets[s])
                class_cost = 0.0
                for label, add in labels.items():
                    goal = label_totals[label] * weights[s]
                    old = observed[s][label]
                    class_cost += (2 * (old - goal) + add) * add / max(1, goal)
                    if old == 0:
                        class_cost -= 2.0
                return overflow, count_cost + class_cost, s

            s = min(range(n_splits), key=score)
            rows[s].extend(group)
            observed[s].update(labels)
        missing = sum(1 for seen in observed for label in classes if not seen[label])
        mismatch = sum(abs(len(r) - n) for r, n in zip(rows, targets))
        error = sum(abs(observed[s][label] - label_totals[label] * weights[s]) / max(1, label_totals[label])
                    for s in range(n_splits) for label in classes)
        rank = (missing, mismatch, error)
        if best is None or rank < best[0]:
            best = rank, rows, observed, 42 + attempt
    rank, rows, observed, seed = best
    if rank[0]:
        raise ValueError("no draft covered all classes; review rare classes and acquisition groups")
    out.mkdir(parents=True)
    details = {}
    for name, selected, image_classes in zip(names, rows, observed):
        paths = sorted(row[0].relative_to(args.images).as_posix() for row in selected)
        (out / (name + ".txt")).write_text("\n".join(paths) + "\n", encoding="utf-8")
        instances = Counter(b[0] for row in selected for b in row[4])
        details[name] = {"images": len(selected), "classes": len(image_classes),
                         "images_by_class": dict(sorted(image_classes.items())),
                         "instances_by_class": dict(sorted(instances.items()))}
    write_json(out / "summary.json", {
        "status": "draft_only_acquisition_groups_unverified", "seed": seed, "layout": args.layout,
        "target_counts": dict(zip(names, targets)),
        "byte_identical_image_groups_kept_together": sum(len(c) > 1 for c in clusters),
        "splits": details, "audit": report,
    })
    print("DRAFT ONLY: " + str(out))
    print(json.dumps({s: {"images": d["images"], "classes": d["classes"]} for s, d in details.items()}))
    print("Review acquisition groups and coordinate conventions before formal export.")


def require_disjoint_image_bytes(groups):
    buckets = defaultdict(list)
    for split, rows in groups.items():
        for row in rows:
            image = row[0]
            buckets[image.stat().st_size].append((split, image))
    for candidates in buckets.values():
        if len({split for split, _ in candidates}) < 2:
            continue
        seen = {}
        for split, image in candidates:
            digest = sha256_file(image)
            if digest in seen and seen[digest][0] != split:
                other_split, other = seen[digest]
                raise ValueError("identical images cross splits: {} ({}) and {} ({})".format(
                    other.name, other_split, image.name, split))
            seen[digest] = (split, image)


def export(args, records, tagged, classes, report):
    if Image is None:
        raise ValueError("export requires Pillow; audit is available without it")
    if not args.coords:
        raise ValueError("export requires --coords; run audit first")
    if len(classes) != args.expected_classes:
        raise ValueError("found {} usable classes, expected {}; review audit first".format(len(classes), args.expected_classes))
    out = destination(args)

    rng = random.Random(42)
    if args.mode == "smoke":
        if tagged:
            train = [r for r in records if r[3] == "train"]
            val = [r for r in records if r[3] == "val"]
        else:
            train, val = list(records), []
        rng.shuffle(train)
        rng.shuffle(val)
        if not val:
            if len(train) < 2:
                raise ValueError("need at least two usable training images for a smoke split")
            count = min(16, max(1, len(train) // 4))
            val, train = train[:count], train[count:]
        groups = {"train": train[:48], "val": val[:16]}
    else:
        if report["issues"]:
            raise ValueError("formal export requires reviewing all audit issues; raw data is untouched")
        names = ("train", "val") if args.layout == "train-val" else ("train", "val", "test")
        if any(r[3] not in names for r in records):
            raise ValueError("formal export requires explicit {} lists for every image; use lists matching --layout".format("/".join(names)))
        groups = {s: [r for r in records if r[3] == s] for s in names}
        for split, rows in groups.items():
            present = {b[0] for row in rows for b in row[4]}
            if present != set(classes):
                raise ValueError("{} is missing classes: {}".format(split, ", ".join(sorted(set(classes) - present))))
    if not all(groups.values()):
        raise ValueError("a required split is empty; review the input paths and split information")
    if args.mode == "formal":
        require_disjoint_image_bytes(groups)

    # A YOLO label is named after its image stem. Refuse a collision rather
    # than renaming source images or overwriting a label.
    total_image_bytes = 0
    for split, rows in groups.items():
        label_names = set()
        for source, *_ in rows:
            label_name = source.relative_to(args.images).with_suffix(".txt").as_posix().casefold()
            if label_name in label_names:
                raise ValueError("same-stem label collision in {}: {}".format(split, label_name))
            label_names.add(label_name)
            total_image_bytes += source.stat().st_size
    existing_parent = next(parent for parent in out.parents if parent.is_dir())
    if shutil.disk_usage(existing_parent).free < total_image_bytes:
        raise ValueError("not enough space for real image copies: need at least {} bytes".format(total_image_bytes))

    names = sorted(classes)
    class_ids = {label: i for i, label in enumerate(names)}
    categories = [{"id": i + 1, "name": label} for i, label in enumerate(names)]
    out.mkdir(parents=True)
    (out / "annotations").mkdir()
    write_json(out / "classes.json", [{"name": n, "yolo_id": i, "coco_id": i + 1} for i, n in enumerate(names)])
    selected = {}
    for split, rows in groups.items():
        image_dir, label_dir = out / "images" / split, out / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        coco_images, annotations = [], []
        selected[split] = []
        for idx, (source, w, h, _, boxes) in enumerate(rows, 1):
            original = source.relative_to(args.images)
            copied_image = image_dir / original
            label_file = label_dir / original.with_suffix(".txt")
            copied_image.parent.mkdir(parents=True, exist_ok=True)
            label_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied_image)
            coco_images.append({"id": idx, "file_name": copied_image.relative_to(out).as_posix(), "width": w, "height": h})
            selected[split].append(original.as_posix())
            lines = []
            for label, x, y, bw, bh in boxes:
                cls = class_ids[label]
                lines.append("{} {:.9f} {:.9f} {:.9f} {:.9f}".format(cls, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
                annotations.append({"id": len(annotations) + 1, "image_id": idx, "category_id": cls + 1, "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(out / "annotations" / ("instances_" + split + ".json"),
                   {"info": {}, "images": coco_images, "annotations": annotations, "categories": categories})
    yaml = ["path: " + json.dumps(out.as_posix(), ensure_ascii=False)]
    yaml += [s + ": images/" + s for s in groups]
    yaml += ["names:"] + ["  {}: {}".format(i, json.dumps(n, ensure_ascii=False)) for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    write_json(out / "audit.json", report)
    write_json(out / "selection.json", {"purpose": args.mode, "layout": args.layout if args.mode == "formal" else "smoke",
                                        "coordinates": args.coords, "image_storage": "copied_original_filenames",
                                        "images": selected})
    print("{} export ready: {}".format(args.mode.upper(), out / "data.yaml"))
    print("Images per split: " + json.dumps({s: len(rows) for s, rows in groups.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "draft", "smoke", "formal"), default="audit")
    parser.add_argument("--layout", choices=("train-val", "train-val-test"), default="train-val-test")
    parser.add_argument("--coords", choices=("zero-half-open", "voc1-inclusive", "zero-inclusive"))
    parser.add_argument("--split-dir", type=Path, help="directory containing train.txt, val.txt and optionally test.txt")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-classes", type=int, default=18)
    args = parser.parse_args()
    try:
        args.images, args.xml = args.images.resolve(strict=True), args.xml.resolve(strict=True)
        if not args.images.is_dir() or not args.xml.is_dir():
            raise ValueError("--images and --xml must be directories")
        records, tagged, classes, report = audit(args.images, args.xml, args.coords, args.split_dir)
        if args.mode == "audit":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.mode == "draft":
            draft_split(args, records, tagged, classes, report)
        else:
            export(args, records, tagged, classes, report)
    except (ValueError, OSError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
