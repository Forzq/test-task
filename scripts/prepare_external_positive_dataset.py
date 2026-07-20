"""Convert public fastener datasets into one two-class YOLO dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


CLASS_NAMES = ("bolt", "nut")
MVTEC_CLASS_MAPPING = {
    7: 1,   # Nut.
    8: 0,   # Bolt.
    9: 1,   # Large nut.
    10: 1,  # Nut.
    11: 1,  # Nut.
}
SPLITS = ("train", "valid", "test")
NPU_CLASS_MAPPING = {
    "bolt_a": 0,  # Bolt head.
    "bolt_b": 1,  # Bolt nut.
    "bolt_c": 0,  # Bolt side.
    "vague": 0,  # Blurred bolt.
}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Axis-aligned target box in pixel coordinates.

    Parameters
    ----------
    class_id : int
        Target class identifier: zero for bolt and one for nut.
    x1, y1, x2, y2 : float
        Top-left and bottom-right pixel coordinates.
    """

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """
        Return the center of the box.

        Returns
        -------
        tuple[float, float]
            Horizontal and vertical center coordinates.
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clipped(self, left: float, top: float, right: float, bottom: float) -> "BoundingBox | None":
        """
        Clip the box to a rectangular crop and translate it to crop coordinates.

        Parameters
        ----------
        left, top, right, bottom : float
            Crop boundaries in source-image pixel coordinates.

        Returns
        -------
        BoundingBox or None
            Translated intersection, or ``None`` when no positive area remains.
        """
        x1 = max(self.x1, left)
        y1 = max(self.y1, top)
        x2 = min(self.x2, right)
        y2 = min(self.y2, bottom)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(self.class_id, x1 - left, y1 - top, x2 - left, y2 - top)


@dataclass(slots=True)
class SplitStatistics:
    """
    Mutable counters for one generated split.

    Parameters
    ----------
    images : int
        Number of generated image files.
    bolts : int
        Number of bolt boxes.
    nuts : int
        Number of nut boxes.
    """

    images: int = 0
    bolts: int = 0
    nuts: int = 0

    def add(self, boxes: Iterable[BoundingBox]) -> None:
        """
        Add one image and its target boxes to the counters.

        Parameters
        ----------
        boxes : Iterable[BoundingBox]
            Boxes written for the generated image.
        """
        materialized = list(boxes)
        self.images += 1
        self.bolts += sum(box.class_id == 0 for box in materialized)
        self.nuts += sum(box.class_id == 1 for box in materialized)


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line parser for dataset preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with repository-local source and output defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict whole-object dataset from MVTec nuts and "
            "Bolts/Washers bolts while excluding incompatible part labels."
        )
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/external_sources"),
        help="Directory containing the three extracted public datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external_positives"),
        help="Generated two-class YOLO dataset directory.",
    )
    parser.add_argument(
        "--minimum-images",
        type=int,
        default=500,
        help="Fail unless at least this many annotated images are generated.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace a previously generated output directory.",
    )
    return parser


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create an empty generated dataset directory.

    Parameters
    ----------
    output : Path
        Destination dataset root.
    replace : bool
        Whether an existing generated directory may be removed.

    Raises
    ------
    FileExistsError
        Raised when output exists and replacement was not requested.
    """
    if output.exists():
        if not replace:
            raise FileExistsError(f"Output already exists; use --replace: {output}")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)


def _write_yolo_label(path: Path, boxes: Iterable[BoundingBox], width: int, height: int) -> None:
    """
    Write pixel-coordinate boxes in normalized YOLO format.

    Parameters
    ----------
    path : Path
        Destination text label path.
    boxes : Iterable[BoundingBox]
        Axis-aligned target boxes.
    width, height : int
        Generated image dimensions in pixels.
    """
    lines: list[str] = []
    for box in boxes:
        center_x = ((box.x1 + box.x2) / 2) / width
        center_y = ((box.y1 + box.y2) / 2) / height
        box_width = (box.x2 - box.x1) / width
        box_height = (box.y2 - box.y1) / height
        values = (center_x, center_y, box_width, box_height)
        if not all(0.0 <= value <= 1.0 for value in values) or box_width <= 0 or box_height <= 0:
            raise ValueError(f"Invalid normalized box for {path}: {box}")
        lines.append(f"{box.class_id} " + " ".join(f"{value:.6f}" for value in values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_example(
    source: Path,
    output: Path,
    split: str,
    target_name: str,
    boxes: list[BoundingBox],
    statistics: dict[str, SplitStatistics],
) -> None:
    """
    Copy one source image and write its converted labels.

    Parameters
    ----------
    source : Path
        Source image path.
    output : Path
        Generated dataset root.
    split : str
        Target split name.
    target_name : str
        Collision-safe generated file name.
    boxes : list[BoundingBox]
        Converted target boxes.
    statistics : dict[str, SplitStatistics]
        Mutable split counters.
    """
    with Image.open(source) as image:
        width, height = image.size
    destination = output / split / "images" / target_name
    shutil.copy2(source, destination)
    _write_yolo_label(output / split / "labels" / f"{Path(target_name).stem}.txt", boxes, width, height)
    statistics[split].add(boxes)


def _oriented_to_axis_aligned(raw_box: list[float], class_id: int, width: int, height: int) -> BoundingBox:
    """
    Enclose an MVTec oriented rectangle with an axis-aligned rectangle.

    Parameters
    ----------
    raw_box : list[float]
        MVTec ``row, col, width, height, phi`` annotation.
    class_id : int
        Converted task class identifier.
    width, height : int
        Source image dimensions.

    Returns
    -------
    BoundingBox
        Clipped axis-aligned enclosing box.
    """
    row, column, box_width, box_height, angle = (float(value) for value in raw_box)
    half_x = abs(math.cos(angle)) * box_width / 2 + abs(math.sin(angle)) * box_height / 2
    half_y = abs(math.sin(angle)) * box_width / 2 + abs(math.cos(angle)) * box_height / 2
    return BoundingBox(
        class_id,
        max(0.0, column - half_x),
        max(0.0, row - half_y),
        min(float(width), column + half_x),
        min(float(height), row + half_y),
    )


def _convert_mvtec(
    source_root: Path, output: Path, statistics: dict[str, SplitStatistics]
) -> dict[str, int]:
    """
    Convert MVTec oriented annotations using the official split files.

    Parameters
    ----------
    source_root : Path
        Extracted MVTec dataset directory.
    output : Path
        Generated dataset root.
    statistics : dict[str, SplitStatistics]
        Mutable split counters.

    Returns
    -------
    dict[str, int]
        Number of converted images per split.
    """
    converted: dict[str, int] = {}
    for split in SPLITS:
        source_split = "val" if split == "valid" else split
        annotation_path = source_root / f"mvtec_screws_{source_split}.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotations: dict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        for image_record in payload["images"]:
            image_id = int(image_record["id"])
            width = int(image_record["width"])
            height = int(image_record["height"])
            boxes = [
                _oriented_to_axis_aligned(
                    annotation["bbox"],
                    MVTEC_CLASS_MAPPING[int(annotation["category_id"])],
                    width,
                    height,
                )
                for annotation in annotations[image_id]
                if int(annotation["category_id"]) in MVTEC_CLASS_MAPPING
            ]
            file_name = str(image_record["file_name"])
            _copy_example(
                source_root / "images" / file_name,
                output,
                split,
                f"mvtec_{file_name}",
                boxes,
                statistics,
            )
        converted[split] = len(payload["images"])
    return converted


def _parse_npu_annotation(path: Path) -> tuple[str, int, int, list[BoundingBox]]:
    """
    Parse one NPU PASCAL VOC annotation.

    Parameters
    ----------
    path : Path
        XML annotation path.

    Returns
    -------
    tuple[str, int, int, list[BoundingBox]]
        Image file name, dimensions, and converted target boxes.
    """
    root = ET.parse(path).getroot()
    file_name = root.findtext("filename") or f"{path.stem}.jpg"
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    boxes: list[BoundingBox] = []
    for item in root.findall("object"):
        source_class = item.findtext("name", "")
        if source_class not in NPU_CLASS_MAPPING:
            raise ValueError(f"Unsupported NPU class {source_class!r} in {path}")
        bounds = item.find("bndbox")
        if bounds is None:
            raise ValueError(f"Missing bndbox in {path}")
        boxes.append(
            BoundingBox(
                NPU_CLASS_MAPPING[source_class],
                float(bounds.findtext("xmin", "0")),
                float(bounds.findtext("ymin", "0")),
                float(bounds.findtext("xmax", "0")),
                float(bounds.findtext("ymax", "0")),
            )
        )
    return file_name, width, height, boxes


def _split_npu(annotation_paths: list[Path], seed: int) -> dict[str, list[Path]]:
    """
    Create deterministic image-level NPU splits with no crop leakage.

    Parameters
    ----------
    annotation_paths : list[Path]
        Complete NPU annotation list.
    seed : int
        Shuffle seed.

    Returns
    -------
    dict[str, list[Path]]
        Train, validation, and test annotation paths.
    """
    shuffled = sorted(annotation_paths)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.8)
    valid_end = train_end + round(len(shuffled) * 0.1)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def _crop_bounds(target: BoundingBox, width: int, height: int) -> tuple[int, int, int, int]:
    """
    Calculate a bounded square crop around one NPU target.

    Parameters
    ----------
    target : BoundingBox
        Target around which the crop is centred.
    width, height : int
        Source image dimensions.

    Returns
    -------
    tuple[int, int, int, int]
        Integer left, top, right, and bottom crop coordinates.
    """
    target_size = max(target.x2 - target.x1, target.y2 - target.y1)
    crop_size = int(min(1600, max(640, target_size * 6)))
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    center_x, center_y = target.center
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    left = min(max(0, left), width - crop_width)
    top = min(max(0, top), height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _write_npu_crop(
    source: Path,
    output: Path,
    name: str,
    crop: tuple[int, int, int, int],
    boxes: list[BoundingBox],
    statistics: dict[str, SplitStatistics],
) -> None:
    """
    Save one target-centred NPU crop with recalculated labels.

    Parameters
    ----------
    source : Path
        Source image path.
    output : Path
        Generated dataset root.
    name : str
        Generated image stem.
    crop : tuple[int, int, int, int]
        Crop coordinates in the source image.
    boxes : list[BoundingBox]
        All source-image boxes.
    statistics : dict[str, SplitStatistics]
        Mutable split counters.
    """
    left, top, right, bottom = crop
    crop_boxes: list[BoundingBox] = []
    for box in boxes:
        center_x, center_y = box.center
        if left <= center_x < right and top <= center_y < bottom:
            clipped = box.clipped(left, top, right, bottom)
            if clipped is not None:
                crop_boxes.append(clipped)
    if not crop_boxes:
        return
    destination = output / "train" / "images" / f"{name}.jpg"
    with Image.open(source) as image:
        image.crop(crop).convert("RGB").save(destination, quality=92, optimize=True)
    _write_yolo_label(
        output / "train" / "labels" / f"{name}.txt",
        crop_boxes,
        right - left,
        bottom - top,
    )
    statistics["train"].add(crop_boxes)


def _convert_npu(
    source_root: Path,
    output: Path,
    statistics: dict[str, SplitStatistics],
    seed: int,
    max_crops: int,
) -> dict[str, int]:
    """
    Convert NPU VOC boxes and add target-centred crops to training only.

    Parameters
    ----------
    source_root : Path
        Extracted NPU dataset directory.
    output : Path
        Generated dataset root.
    statistics : dict[str, SplitStatistics]
        Mutable split counters.
    seed : int
        Deterministic split seed.
    max_crops : int
        Maximum number of derived training crops.

    Returns
    -------
    dict[str, int]
        Original split counts and generated crop count.
    """
    voc_root = source_root / "VOCFormat"
    splits = _split_npu(list((voc_root / "Annotations").glob("*.xml")), seed)
    crop_count = 0
    crop_keys: set[tuple[str, int, int, int, int]] = set()
    for split, paths in splits.items():
        for annotation_path in paths:
            file_name, width, height, boxes = _parse_npu_annotation(annotation_path)
            source_image = voc_root / "JPEGImages" / file_name
            if not source_image.is_file():
                source_image = voc_root / "JPEGImages" / f"{annotation_path.stem}.jpg"
            _copy_example(
                source_image,
                output,
                split,
                f"npu_{source_image.name}",
                boxes,
                statistics,
            )
            if split != "train" or crop_count >= max_crops:
                continue
            for target_index, target in enumerate(boxes):
                if crop_count >= max_crops:
                    break
                crop = _crop_bounds(target, width, height)
                if crop == (0, 0, width, height):
                    continue
                key = (source_image.name, *crop)
                if key in crop_keys:
                    continue
                crop_keys.add(key)
                name = f"npu_crop_{annotation_path.stem}_{target_index:03d}"
                _write_npu_crop(source_image, output, name, crop, boxes, statistics)
                crop_count += 1
    return {
        "train_original": len(splits["train"]),
        "valid_original": len(splits["valid"]),
        "test_original": len(splits["test"]),
        "train_crops": crop_count,
    }


def _parse_yolo_boxes(path: Path, width: int, height: int) -> list[BoundingBox]:
    """
    Read bolt boxes from the Bolts/Washers YOLO source.

    Parameters
    ----------
    path : Path
        Source YOLO label path.
    width, height : int
        Source image dimensions.

    Returns
    -------
    list[BoundingBox]
        Class-zero bolt boxes; bottle and washer boxes are intentionally excluded.
    """
    boxes: list[BoundingBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values or int(values[0]) != 0:
            continue
        coordinates = [float(value) for value in values[1:]]
        if len(coordinates) == 4:
            center_x, center_y, box_width, box_height = coordinates
            normalized = (
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            )
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            x_values = coordinates[0::2]
            y_values = coordinates[1::2]
            normalized = (min(x_values), min(y_values), max(x_values), max(y_values))
        else:
            raise ValueError(f"Malformed YOLO box or polygon in {path}")
        x1, y1, x2, y2 = (min(1.0, max(0.0, value)) for value in normalized)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(BoundingBox(0, x1 * width, y1 * height, x2 * width, y2 * height))
    return boxes


def _convert_bolts_and_washers(
    source_root: Path, output: Path, statistics: dict[str, SplitStatistics]
) -> dict[str, int]:
    """
    Convert the MIT Bolts/Washers dataset while discarding non-target boxes.

    Parameters
    ----------
    source_root : Path
        Extracted source dataset directory.
    output : Path
        Generated dataset root.
    statistics : dict[str, SplitStatistics]
        Mutable split counters.

    Returns
    -------
    dict[str, int]
        Number of converted images per split.
    """
    source_names = {"train": "train", "valid": "valid", "test": "test"}
    converted: dict[str, int] = {}
    for split, source_split in source_names.items():
        image_directory = source_root / source_split / "images"
        count = 0
        for image_path in sorted(image_directory.glob("*.jpg")):
            with Image.open(image_path) as image:
                width, height = image.size
            label_path = source_root / source_split / "labels" / f"{image_path.stem}.txt"
            boxes = _parse_yolo_boxes(label_path, width, height)
            _copy_example(
                image_path,
                output,
                split,
                f"bw_{image_path.name}",
                boxes,
                statistics,
            )
            count += 1
        converted[split] = count
    return converted


def _write_metadata(
    output: Path,
    statistics: dict[str, SplitStatistics],
    source_counts: dict[str, dict[str, int]],
) -> None:
    """
    Write YOLO configuration, provenance, and license notes.

    Parameters
    ----------
    output : Path
        Generated dataset root.
    statistics : dict[str, SplitStatistics]
        Final split counters.
    source_counts : dict[str, dict[str, int]]
        Per-source conversion counts.
    """
    data_yaml = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": list(CLASS_NAMES),
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    sources = {
        "npu_bolt_excluded": {
            "url": "https://www.kaggle.com/datasets/yartinz/npu-bolt",
            "license": "CC0: Public Domain",
            "reason": (
                "Annotations describe bolt head, side, blur, and nut parts rather "
                "than one complete physical bolt per box."
            ),
        },
        "mvtec_screws": {
            "url": "https://www.mvtec.com/research-teaching/datasets/mvtec-screws",
            "license": "CC BY-NC-SA 4.0 (non-commercial)",
            "mapping": (
                "type_008 -> bolt; type_007/type_009/type_010/type_011 -> nut; "
                "all screw types ignored as background"
            ),
        },
        "bolts_and_washers": {
            "url": "https://www.kaggle.com/datasets/ahmedmohamedab/bolts-and-washers",
            "license": "MIT",
            "mapping": "Bolt -> bolt; Bottle and Washer boxes discarded",
        },
    }
    manifest = {
        "classes": list(CLASS_NAMES),
        "splits": {split: asdict(summary) for split, summary in statistics.items()},
        "source_counts": source_counts,
        "sources": sources,
        "notes": [
            "The ontology requires one complete physical bolt or nut per box.",
            "NPU is excluded because its part-level boxes cannot be merged reliably.",
            "MVTec oriented boxes are converted to enclosing axis-aligned boxes.",
            "MVTec screws and Bolts/Washers bottles and washers are hard-negative background.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def prepare(arguments: argparse.Namespace) -> dict[str, SplitStatistics]:
    """
    Build and validate the complete external positive dataset.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command-line options.

    Returns
    -------
    dict[str, SplitStatistics]
        Final per-split image and box statistics.

    Raises
    ------
    ValueError
        Raised when sources are absent or the requested image minimum is not met.
    """
    source_root = arguments.sources.resolve()
    output = arguments.output.resolve()
    expected = {
        "mvtec": source_root / "mvtec_screws_nuts",
        "bolts_and_washers": source_root / "bolts_and_washers",
    }
    missing = [str(path) for path in expected.values() if not path.is_dir()]
    if missing:
        raise ValueError(f"Extracted source directories are missing: {', '.join(missing)}")
    if arguments.minimum_images <= 0:
        raise ValueError("minimum-images must be positive")

    _prepare_output(output, arguments.replace)
    statistics = {split: SplitStatistics() for split in SPLITS}
    source_counts = {
        "mvtec": _convert_mvtec(expected["mvtec"], output, statistics),
        "bolts_and_washers": _convert_bolts_and_washers(
            expected["bolts_and_washers"], output, statistics
        ),
    }
    total_images = sum(item.images for item in statistics.values())
    if total_images < arguments.minimum_images:
        raise ValueError(
            f"Generated only {total_images} images; required at least {arguments.minimum_images}"
        )
    _write_metadata(output, statistics, source_counts)
    return statistics


def main() -> None:
    """
    Generate the external YOLO dataset and print verified statistics.
    """
    arguments = build_parser().parse_args()
    statistics = prepare(arguments)
    for split, summary in statistics.items():
        print(
            f"{split}: {summary.images} images, "
            f"{summary.bolts} bolt boxes, {summary.nuts} nut boxes"
        )
    print(f"Generated dataset: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
