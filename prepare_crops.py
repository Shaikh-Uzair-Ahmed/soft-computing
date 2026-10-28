"""Crop RDD2022 damage boxes and build a balanced stratified split.

The Czech training images are annotated in PASCAL VOC XML. Test images have
no boxes, so only the training split of the country pack is used. Each kept
box becomes one crop of a damage class the RDD2022 paper evaluates:
D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.
"""

from __future__ import annotations

import argparse
import csv
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLASSES = ("D00", "D10", "D20", "D40")
CLASS_NAMES = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}
CZECH_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/"
    "RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip"
)
HF_REPO = "dronefreak/RDD2022"
YOLO_TO_CLASS = {"0": "D00", "1": "D10", "2": "D20", "3": "D40"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def download_czech(dest_zip: Path) -> None:
    """Download the Czech country pack if it is not already on disk."""
    if dest_zip.exists() and dest_zip.stat().st_size > 0:
        print(f"Using existing archive {dest_zip}")
        return
    import urllib.request

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Czech RDD2022 subset to {dest_zip}")
    urllib.request.urlretrieve(CZECH_URL, dest_zip)


def extract_zip(dest_zip: Path, dest_dir: Path) -> Path:
    marker = dest_dir / ".extracted"
    if marker.exists():
        print(f"Using extracted data in {dest_dir}")
        return dest_dir
    print(f"Extracting {dest_zip} to {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip) as archive:
        archive.extractall(dest_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return dest_dir


def _local_name(element: ET.Element) -> str:
    return element.tag.split("}")[-1].lower()


def _find_text(parent: ET.Element, name: str) -> str | None:
    for child in parent.iter():
        if _local_name(child) == name and child.text:
            return child.text.strip()
    return None


def parse_boxes(xml_path: Path) -> tuple[str | None, list[tuple[str, int, int, int, int]]]:
    """Return the image filename and VOC boxes whose labels are in CLASSES."""
    root = ET.parse(xml_path).getroot()
    filename = _find_text(root, "filename")
    boxes: list[tuple[str, int, int, int, int]] = []
    for obj in root.iter():
        if _local_name(obj) != "object":
            continue
        label = _find_text(obj, "name") or _find_text(obj, "label")
        if label not in CLASSES:
            continue
        bnd = None
        for child in obj.iter():
            if _local_name(child) == "bndbox":
                bnd = child
                break
        if bnd is None:
            continue
        try:
            xmin = int(float(_find_text(bnd, "xmin") or "nan"))
            ymin = int(float(_find_text(bnd, "ymin") or "nan"))
            xmax = int(float(_find_text(bnd, "xmax") or "nan"))
            ymax = int(float(_find_text(bnd, "ymax") or "nan"))
        except ValueError:
            continue
        boxes.append((label, xmin, ymin, xmax, ymax))
    return filename, boxes


def index_images(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def locate_image(xml_path: Path, filename: str | None, index: dict[str, Path]) -> Path | None:
    names = []
    if filename:
        names.append(filename)
    names.append(xml_path.with_suffix(".jpg").name)
    for name in names:
        sibling = xml_path.parent / name
        if sibling.exists():
            return sibling
        parallel = xml_path.parent.parent / "images" / name
        if parallel.exists():
            return parallel
        stem = Path(name).stem
        if stem in index:
            return index[stem]
    return None


def is_training_annotation(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "test" not in parts


def collect_crops(data_root: Path, min_side: int) -> list[dict]:
    images = index_images(data_root)
    records: list[dict] = []
    xml_paths = [path for path in data_root.rglob("*.xml") if is_training_annotation(path)]
    if not xml_paths:
        raise FileNotFoundError(f"No training XML annotations found under {data_root}")
    print(f"Found {len(xml_paths)} training annotation files")
    for xml_path in xml_paths:
        filename, boxes = parse_boxes(xml_path)
        if not boxes:
            continue
        image_path = locate_image(xml_path, filename, images)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for label, xmin, ymin, xmax, ymax in boxes:
                xmin = max(0, min(xmin, width - 1))
                ymin = max(0, min(ymin, height - 1))
                xmax = max(0, min(xmax, width))
                ymax = max(0, min(ymax, height))
                box_w = xmax - xmin
                box_h = ymax - ymin
                if box_w < min_side or box_h < min_side:
                    continue
                records.append(
                    {
                        "image_path": image_path,
                        "label": label,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "aspect_ratio": box_w / box_h,
                        "relative_area": (box_w * box_h) / (width * height),
                    }
                )
    if not records:
        raise RuntimeError("Annotations were found, but no usable damage boxes were kept")
    return records


def balanced_sample(records: list[dict], per_class: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)
    rng = random.Random(seed)
    chosen: list[dict] = []
    for label in CLASSES:
        rows = grouped.get(label, [])
        if not rows:
            print(f"{label} ({CLASS_NAMES[label]}): no boxes in this country subset, skipping")
            continue
        rng.shuffle(rows)
        kept = rows[:per_class]
        print(f"{label} ({CLASS_NAMES[label]}): {len(rows)} boxes, keeping {len(kept)}")
        if len(kept) < 2:
            raise RuntimeError(f"Class {label} has fewer than 2 usable boxes")
        chosen.extend(kept)
    if len({row["label"] for row in chosen}) < 2:
        raise RuntimeError("Need at least two damage classes to train a classifier")
    return chosen


def stratified_split(records: list[dict], test_size: float, seed: int) -> list[dict]:
    """Assign train/test labels with the same class proportions."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)
    rng = random.Random(seed)
    for label, rows in grouped.items():
        rng.shuffle(rows)
        n_test = int(round(len(rows) * test_size))
        n_test = min(max(n_test, 1), len(rows) - 1)
        for index, row in enumerate(rows):
            row["split"] = "test" if index < n_test else "train"
    return records


def save_crops(records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "label",
                "split",
                "aspect_ratio",
                "relative_area",
            ],
        )
        writer.writeheader()
        counters = defaultdict(int)
        for record in records:
            counters[record["label"]] += 1
            name = f"{record['split']}_{record['label']}_{counters[record['label']]:04d}.jpg"
            dest = out_dir / record["split"] / record["label"] / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(record["image_path"]) as image:
                crop = image.convert("RGB").crop(
                    (record["xmin"], record["ymin"], record["xmax"], record["ymax"])
                )
                crop.save(dest, quality=90)
            writer.writerow(
                {
                    "path": dest.relative_to(out_dir).as_posix(),
                    "label": record["label"],
                    "split": record["split"],
                    "aspect_ratio": f"{record['aspect_ratio']:.6f}",
                    "relative_area": f"{record['relative_area']:.6f}",
                }
            )
    print(f"Wrote {len(records)} crops and {csv_path}")
    return csv_path


def parse_yolo_boxes(label_path: Path) -> list[tuple[str, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        label = YOLO_TO_CLASS.get(parts[0])
        if label is None:
            continue
        cx, cy, width, height = (float(value) for value in parts[1:5])
        boxes.append((label, cx, cy, width, height))
    return boxes


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part.split("<", 1)[1].split(">", 1)[0]
    return None


def iter_repo_tree(repo_path: str):
    import json
    import urllib.request

    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/{repo_path}?limit=1000"
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": "rdd-anfis"})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
            url = _next_link(response.headers.get("Link"))
        yield from page


def list_label_paths(prefix: str) -> list[str]:
    shards = [
        item["path"]
        for item in iter_repo_tree("data/labels/train")
        if item.get("type") == "directory" or Path(item["path"]).name.startswith("shard_")
    ]
    found: list[str] = []
    for shard in shards:
        for item in iter_repo_tree(shard):
            name = Path(item["path"]).name
            if name.startswith(prefix) and name.endswith(".txt"):
                found.append(item["path"])
    if not found:
        raise RuntimeError(f"Could not list {prefix} labels on the Hugging Face mirror")
    print(f"Listed {len(found)} {prefix} label files")
    return found


def download_repo_files(paths: list[str], root: Path, allow_empty: bool) -> None:
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def needed(repo_path: str) -> bool:
        dest = root / repo_path
        if not dest.exists():
            return True
        return (not allow_empty) and dest.stat().st_size == 0

    pending = [repo_path for repo_path in paths if needed(repo_path)]
    if not pending:
        return

    def fetch(repo_path: str) -> str:
        import time

        dest = root / repo_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{repo_path}"
        temporary = dest.with_suffix(dest.suffix + ".part")
        delay = 1.0
        last_error: Exception | None = None
        for _ in range(6):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "rdd-anfis"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        handle.write(chunk)
                temporary.replace(dest)
                return repo_path
            except Exception as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                time.sleep(delay)
                delay = min(delay * 2, 20)
        raise RuntimeError(f"Failed to download {repo_path}") from last_error

    workers = min(3, len(pending))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, repo_path) for repo_path in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 200 == 0 or done == len(pending):
                print(f"Downloaded {done}/{len(pending)}")


def download_hf_country(dest: Path, prefix: str, per_class: int, min_side: int, seed: int) -> Path:
    """Download one country's RDD2022 training labels until each class has enough boxes.

    The country zip linked from the Sekilab repository currently returns HTTP
    403. These are the same training photographs, redistributed with YOLO
    boxes (class 0 D00, 1 D10, 2 D20, 3 D40). The Czech training labels contain
    no potholes, so that class is skipped when this country is used.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths = list_label_paths(prefix)
    random.Random(seed).shuffle(paths)
    print(f"Downloading {prefix} training labels from {HF_REPO}")
    batch_size = 120
    for start in range(0, len(paths), batch_size):
        download_repo_files(paths[start : start + batch_size], dest, allow_empty=True)
        counts = {label: 0 for label in CLASSES}
        for record in collect_yolo_candidates(dest, min_side, prefix):
            counts[record["label"]] += 1
        print("Boxes so far: " + ", ".join(f"{label}={counts[label]}" for label in CLASSES))
        if all(counts[label] >= per_class for label in CLASSES):
            break
    return dest


def image_path_for_label(label_path: Path, root: Path) -> Path:
    relative = label_path.relative_to(root).as_posix()
    image_rel = relative.replace("/labels/", "/images/").replace(".txt", ".jpg")
    return root / image_rel


def collect_yolo_candidates(root: Path, min_side: int, prefix: str) -> list[dict]:
    label_paths = [
        path
        for path in root.rglob(f"{prefix}*.txt")
        if path.name.startswith(prefix) and "labels" in path.parts and "train" in path.parts
    ]
    if not label_paths:
        return []
    records: list[dict] = []
    for label_path in label_paths:
        for label, cx, cy, norm_w, norm_h in parse_yolo_boxes(label_path):
            if norm_w <= 0 or norm_h <= 0 or min_side < 0:
                continue
            records.append(
                {
                    "label_path": label_path,
                    "image_path": image_path_for_label(label_path, root),
                    "repo_image": image_path_for_label(label_path, root).relative_to(root).as_posix(),
                    "label": label,
                    "cx": cx,
                    "cy": cy,
                    "norm_w": norm_w,
                    "norm_h": norm_h,
                }
            )
    return records


def download_selected_images(root: Path, records: list[dict]) -> None:
    missing = sorted({record["repo_image"] for record in records if not record["image_path"].exists()})
    if not missing:
        print("Selected training images are already on disk")
        return
    print(f"Downloading {len(missing)} training images")
    download_repo_files(missing, root, allow_empty=False)


def pixel_boxes(records: list[dict], min_side: int) -> list[dict]:
    ready: list[dict] = []
    for record in records:
        image_path = record["image_path"]
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        box_w = record["norm_w"] * width
        box_h = record["norm_h"] * height
        if box_w < min_side or box_h < min_side:
            continue
        xmin = int(round((record["cx"] - record["norm_w"] / 2) * width))
        ymin = int(round((record["cy"] - record["norm_h"] / 2) * height))
        xmax = int(round((record["cx"] + record["norm_w"] / 2) * width))
        ymax = int(round((record["cy"] + record["norm_h"] / 2) * height))
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(xmin + 1, min(xmax, width))
        ymax = max(ymin + 1, min(ymax, height))
        ready.append(
            {
                "image_path": image_path,
                "label": record["label"],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "aspect_ratio": (xmax - xmin) / (ymax - ymin),
                "relative_area": ((xmax - xmin) * (ymax - ymin)) / (width * height),
            }
        )
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/czech"),
        help="Extracted Czech RDD2022 directory, or the destination used with --download",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/crops"))
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-side", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--country",
        default="Czech",
        help="RDD2022 country prefix to download. Czech has no pothole boxes, so D40 is skipped for that country.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the Czech RDD2022 training subset (Hugging Face mirror; the country zip often returns 403)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data/RDD2022_Czech.zip"),
    )
    args = parser.parse_args()
    data_root = args.data_root
    if args.zip_path.exists() and zipfile.is_zipfile(args.zip_path):
        data_root = extract_zip(args.zip_path, args.data_root)
        records = collect_crops(data_root, args.min_side)
    elif args.download or any(args.data_root.rglob(f"{args.country}_*.txt")):
        prefix = f"{args.country}_"
        if args.download:
            data_root = download_hf_country(args.data_root, prefix, args.per_class, args.min_side, args.seed)
        candidates = collect_yolo_candidates(data_root, args.min_side, prefix)
        candidates = balanced_sample(candidates, args.per_class, args.seed)
        download_selected_images(data_root, candidates)
        records = pixel_boxes(candidates, args.min_side)
        print(f"Usable crops after pixel clipping: {len(records)}")
    else:
        records = collect_crops(data_root, args.min_side)
    records = balanced_sample(records, args.per_class, args.seed)
    records = stratified_split(records, args.test_size, args.seed)
    save_crops(records, args.out_dir)


if __name__ == "__main__":
    main()
