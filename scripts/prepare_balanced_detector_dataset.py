"""Build a source- and scale-balanced view of additional detector data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")
SOURCE_NAMES = (
    "fastener_object_detection",
    "screw_nut_bolt",
    "stresstest_tool_tray",
)
GENERATED_MARKER = ".generated-balanced-detector-dataset"


@dataclass(frozen=True, slots=True)
class Quota:
    """
    Limit class-composition and negative scenes from one source split.

    Parameters
    ----------
    bolt_only : int
        Maximum images containing bolts but no nuts.
    nut_only : int
        Maximum images containing nuts but no bolts.
    mixed : int
        Maximum images containing both target classes.
    negative : int
        Maximum empty-label background images.
    """

    bolt_only: int
    nut_only: int
    mixed: int
    negative: int


DEFAULT_QUOTAS: dict[str, dict[str, Quota]] = {
    "train": {
        "fastener_object_detection": Quota(
            bolt_only=10, nut_only=10, mixed=10, negative=200
        ),
        "screw_nut_bolt": Quota(
            bolt_only=100, nut_only=220, mixed=0, negative=100
        ),
        "stresstest_tool_tray": Quota(
            bolt_only=999, nut_only=999, mixed=999, negative=200
        ),
    },
    "valid": {
        "fastener_object_detection": Quota(
            bolt_only=10, nut_only=10, mixed=10, negative=40
        ),
        "screw_nut_bolt": Quota(
            bolt_only=20, nut_only=20, mixed=0, negative=20
        ),
        "stresstest_tool_tray": Quota(
            bolt_only=999, nut_only=999, mixed=999, negative=50
        ),
    },
    "test": {
        "fastener_object_detection": Quota(
            bolt_only=10, nut_only=10, mixed=10, negative=40
        ),
        "screw_nut_bolt": Quota(
            bolt_only=20, nut_only=20, mixed=0, negative=20
        ),
        "stresstest_tool_tray": Quota(
            bolt_only=999, nut_only=999, mixed=999, negative=40
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """
    Describe one normalized source image and its target annotations.

    Parameters
    ----------
    image_path : Path
        Input image path.
    label_path : Path
        Matching strict YOLO label path.
    split : str
        Dataset split.
    source : str
        Source identifier encoded in the normalized filename.
    bolt_boxes : int
        Number of bolt annotations.
    nut_boxes : int
        Number of nut annotations.
    median_area : float
        Median normalized box area, or zero for a negative image.
    """

    image_path: Path
    label_path: Path
    split: str
    source: str
    bolt_boxes: int
    nut_boxes: int
    median_area: float

    @property
    def object_count(self) -> int:
        """Return the number of target objects in this image."""
        return self.bolt_boxes + self.nut_boxes

    @property
    def is_positive(self) -> bool:
        """Return whether the image contains at least one target box."""
        return self.object_count > 0


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for balanced dataset preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with separate source, output, and combined-YAML paths.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stratify and cap additional Roboflow scenes so dense tiny-object "
            "sources cannot dominate detector training."
        )
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/roboflow_additions")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/roboflow_balanced")
    )
    parser.add_argument(
        "--combined-input", type=Path, default=Path("data/combined.yaml")
    )
    parser.add_argument(
        "--combined-output", type=Path, default=Path("data/combined_balanced.yaml")
    )
    parser.add_argument(
        "--quota-config",
        type=Path,
        help="Optional JSON file overriding all default per-split source quotas.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    return parser


def _source_name(path: Path) -> str:
    """
    Extract the source identifier from a normalized image filename.

    Parameters
    ----------
    path : Path
        Image whose name starts with ``<source>__``.

    Returns
    -------
    str
        Validated source identifier.
    """
    source = path.name.split("__", 1)[0]
    if source not in SOURCE_NAMES:
        raise ValueError(f"Unknown normalized source prefix in {path.name}")
    return source


def _read_record(image_path: Path, label_path: Path, split: str) -> ImageRecord:
    """
    Parse and validate one strict two-class YOLO annotation.

    Parameters
    ----------
    image_path : Path
        Normalized source image.
    label_path : Path
        Matching YOLO label.
    split : str
        Canonical split name.

    Returns
    -------
    ImageRecord
        Parsed class counts and scale summary.
    """
    if not label_path.is_file():
        raise ValueError(f"Missing label for {image_path}")
    counts: Counter[int] = Counter()
    areas: list[float] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5 or values[0] not in {"0", "1"}:
            raise ValueError(f"Expected strict YOLO box at {label_path}:{line_number}")
        coordinates = [float(value) for value in values[1:]]
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"Coordinate outside [0, 1] at {label_path}:{line_number}")
        counts[int(values[0])] += 1
        areas.append(coordinates[2] * coordinates[3])
    return ImageRecord(
        image_path=image_path,
        label_path=label_path,
        split=split,
        source=_source_name(image_path),
        bolt_boxes=counts[0],
        nut_boxes=counts[1],
        median_area=median(areas) if areas else 0.0,
    )


def _records(source_root: Path) -> dict[str, dict[str, list[ImageRecord]]]:
    """
    Enumerate normalized images grouped by split and source.

    Parameters
    ----------
    source_root : Path
        Strict ``data/roboflow_additions`` dataset root.

    Returns
    -------
    dict[str, dict[str, list[ImageRecord]]]
        Records keyed first by split and then source.
    """
    grouped: dict[str, dict[str, list[ImageRecord]]] = {
        split: {source: [] for source in SOURCE_NAMES} for split in SPLITS
    }
    for split in SPLITS:
        image_root = source_root / split / "images"
        label_root = source_root / split / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            raise ValueError(f"Missing normalized split below {source_root / split}")
        for image_path in sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label_path = label_root / f"{image_path.stem}.txt"
            record = _read_record(image_path, label_path, split)
            grouped[split][record.source].append(record)
    return grouped


def _bucket(record: ImageRecord) -> tuple[str, str, str]:
    """
    Assign a positive image to class, density, and scale strata.

    Parameters
    ----------
    record : ImageRecord
        Positive normalized source record.

    Returns
    -------
    tuple[str, str, str]
        Class-composition, object-density, and median-scale bucket.
    """
    classes = _class_composition(record)

    if record.object_count <= 2:
        density = "sparse"
    elif record.object_count <= 10:
        density = "medium"
    elif record.object_count <= 20:
        density = "dense"
    else:
        density = "very_dense"

    if record.median_area < 0.003:
        scale = "tiny"
    elif record.median_area < 0.02:
        scale = "small"
    elif record.median_area < 0.15:
        scale = "medium"
    else:
        scale = "large"
    return classes, density, scale


def _class_composition(record: ImageRecord) -> str:
    """
    Return the target-class composition of one positive image.

    Parameters
    ----------
    record : ImageRecord
        Positive source record.

    Returns
    -------
    str
        ``bolt_only``, ``nut_only``, or ``mixed``.
    """
    if record.bolt_boxes and record.nut_boxes:
        return "mixed"
    if record.bolt_boxes:
        return "bolt_only"
    if record.nut_boxes:
        return "nut_only"
    raise ValueError("Class composition is undefined for a negative image")


def _stable_seed(seed: int, *parts: str) -> int:
    """
    Derive a platform-independent random seed from textual components.

    Parameters
    ----------
    seed : int
        User-supplied base seed.
    *parts : str
        Split, source, and stratum identifiers.

    Returns
    -------
    int
        Deterministic 64-bit seed.
    """
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _stratified_positive_sample(
    records: list[ImageRecord], quota: int, seed: int, split: str, source: str
) -> list[ImageRecord]:
    """
    Select positives round-robin across class, density, and scale strata.

    Parameters
    ----------
    records : list[ImageRecord]
        Positive candidates from one source split.
    quota : int
        Maximum retained candidates.
    seed : int
        Reproducibility seed.
    split : str
        Split name used to derive independent shuffles.
    source : str
        Source name used to derive independent shuffles.

    Returns
    -------
    list[ImageRecord]
        Reproducibly selected positive records.
    """
    if len(records) <= quota:
        return sorted(records, key=lambda record: record.image_path.name)
    groups: defaultdict[tuple[str, str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[_bucket(record)].append(record)
    queues: dict[tuple[str, str, str], deque[ImageRecord]] = {}
    for bucket, values in sorted(groups.items()):
        shuffled = sorted(values, key=lambda record: record.image_path.name)
        random.Random(
            _stable_seed(seed, split, source, *bucket)
        ).shuffle(shuffled)
        queues[bucket] = deque(shuffled)

    selected: list[ImageRecord] = []
    active = list(sorted(queues))
    while active and len(selected) < quota:
        next_active: list[tuple[str, str, str]] = []
        for bucket in active:
            if queues[bucket] and len(selected) < quota:
                selected.append(queues[bucket].popleft())
            if queues[bucket]:
                next_active.append(bucket)
        active = next_active
    return selected


def _negative_sample(
    records: list[ImageRecord], quota: int, seed: int, split: str, source: str
) -> list[ImageRecord]:
    """
    Select reproducible background images from one source split.

    Parameters
    ----------
    records : list[ImageRecord]
        Empty-label candidates.
    quota : int
        Maximum retained candidates.
    seed : int
        Reproducibility seed.
    split : str
        Split name.
    source : str
        Source name.

    Returns
    -------
    list[ImageRecord]
        Deterministically shuffled negative sample.
    """
    values = sorted(records, key=lambda record: record.image_path.name)
    random.Random(_stable_seed(seed, split, source, "negative")).shuffle(values)
    return values[:quota]


def _load_quotas(path: Path | None) -> dict[str, dict[str, Quota]]:
    """
    Load optional JSON quotas or return immutable defaults.

    Parameters
    ----------
    path : Path, optional
        JSON document containing every split and source quota.

    Returns
    -------
    dict[str, dict[str, Quota]]
        Validated per-split source quotas.
    """
    if path is None:
        return DEFAULT_QUOTAS
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, Quota]] = {}
    for split in SPLITS:
        output[split] = {}
        for source in SOURCE_NAMES:
            values = payload[split][source]
            quota = Quota(
                bolt_only=int(values["bolt_only"]),
                nut_only=int(values["nut_only"]),
                mixed=int(values["mixed"]),
                negative=int(values["negative"]),
            )
            if any(value < 0 for value in asdict(quota).values()):
                raise ValueError("Quotas cannot be negative")
            output[split][source] = quota
    return output


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create a safely replaceable generated dataset directory.

    Parameters
    ----------
    output : Path
        Generated balanced dataset root.
    replace : bool
        Whether an existing marked output may be rebuilt.
    """
    if output.exists():
        if not replace:
            raise ValueError(f"Output exists: {output}; pass --replace")
        if not (output / GENERATED_MARKER).is_file():
            raise ValueError(f"Refusing to replace unmarked directory: {output}")
        resolved = output.resolve()
        project = Path.cwd().resolve()
        if resolved == project or project not in resolved.parents:
            raise ValueError(f"Refusing to remove path outside the project: {resolved}")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)
    (output / GENERATED_MARKER).write_text("generated\n", encoding="utf-8")


def _link_or_copy(source: Path, destination: Path) -> None:
    """
    Hard-link a selected file and fall back to metadata-preserving copy.

    Parameters
    ----------
    source : Path
        Existing image or label.
    destination : Path
        Generated dataset path.
    """
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _record_summary(records: list[ImageRecord]) -> dict[str, Any]:
    """
    Summarize selected or available records for a manifest.

    Parameters
    ----------
    records : list[ImageRecord]
        Records to aggregate.

    Returns
    -------
    dict
        Image, class-box, negative, density, and scale counts.
    """
    summary: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    for record in records:
        summary["images"] += 1
        summary["positive_images"] += int(record.is_positive)
        summary["negative_images"] += int(not record.is_positive)
        summary["bolt_boxes"] += record.bolt_boxes
        summary["nut_boxes"] += record.nut_boxes
        if record.is_positive:
            strata["/".join(_bucket(record))] += 1
    return {**dict(summary), "positive_strata": dict(sorted(strata.items()))}


def _write_combined_yaml(input_path: Path, output_path: Path, balanced_root: Path) -> None:
    """
    Replace only the full Roboflow source references in a copied combined YAML.

    Parameters
    ----------
    input_path : Path
        Existing combined training YAML.
    output_path : Path
        Separate balanced composition YAML.
    balanced_root : Path
        Generated balanced source root.
    """
    document: dict[str, Any] = yaml.safe_load(input_path.read_text(encoding="utf-8")) or {}
    for yaml_key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
        current = document.get(yaml_key)
        values = current if isinstance(current, list) else [current]
        replacement = Path(
            os.path.relpath(
                (balanced_root / split / "images").resolve(),
                start=output_path.parent.resolve(),
            )
        ).as_posix()
        replaced = False
        updated: list[str] = []
        for value in values:
            if value is None:
                continue
            text = str(value).replace("\\", "/")
            if "roboflow_additions/" in text:
                if not replaced:
                    updated.append(replacement)
                    replaced = True
                continue
            updated.append(str(value))
        if not replaced:
            raise ValueError(f"No roboflow_additions reference found in {yaml_key}")
        document[yaml_key] = updated
    document["names"] = ["bolt", "nut"]
    output_path.write_text(
        "# Source-balanced training composition; external benchmark is excluded.\n"
        + yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def prepare(
    source_root: Path,
    output: Path,
    combined_input: Path,
    combined_output: Path,
    quotas: dict[str, dict[str, Quota]],
    seed: int,
    replace: bool,
) -> dict[str, Any]:
    """
    Generate the balanced source, manifest, and separate combined YAML.

    Parameters
    ----------
    source_root : Path
        Full normalized Roboflow addition root.
    output : Path
        Balanced selected-source root.
    combined_input : Path
        Current complete composition YAML.
    combined_output : Path
        Separate composition YAML to generate.
    quotas : dict[str, dict[str, Quota]]
        Per-split source image quotas.
    seed : int
        Deterministic selection seed.
    replace : bool
        Whether a previous generated output may be rebuilt.

    Returns
    -------
    dict
        Reproducibility and selection manifest.
    """
    grouped = _records(source_root)
    _prepare_output(output, replace)
    manifest_sources: dict[str, dict[str, Any]] = {}
    selected_totals: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        manifest_sources[split] = {}
        for source in SOURCE_NAMES:
            candidates = grouped[split][source]
            positives = [record for record in candidates if record.is_positive]
            negatives = [record for record in candidates if not record.is_positive]
            quota = quotas[split][source]
            selected: list[ImageRecord] = []
            for composition in ("bolt_only", "nut_only", "mixed"):
                composition_records = [
                    record
                    for record in positives
                    if _class_composition(record) == composition
                ]
                selected.extend(
                    _stratified_positive_sample(
                        composition_records,
                        getattr(quota, composition),
                        seed,
                        split,
                        f"{source}:{composition}",
                    )
                )
            selected.extend(
                _negative_sample(negatives, quota.negative, seed, split, source)
            )
            selected = sorted(selected, key=lambda record: record.image_path.name)
            selected_totals[split].extend(selected)
            for record in selected:
                _link_or_copy(
                    record.image_path, output / split / "images" / record.image_path.name
                )
                _link_or_copy(
                    record.label_path, output / split / "labels" / record.label_path.name
                )
            manifest_sources[split][source] = {
                "quota": asdict(quota),
                "available": _record_summary(candidates),
                "selected": _record_summary(selected),
            }

    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["bolt", "nut"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_combined_yaml(combined_input, combined_output, output)
    manifest = {
        "description": (
            "Source-, density-, class-, and scale-balanced subset of "
            "data/roboflow_additions."
        ),
        "source": str(source_root.resolve()),
        "combined_input": str(combined_input.resolve()),
        "combined_output": str(combined_output.resolve()),
        "seed": seed,
        "selection_policy": (
            "Positive images are sampled round-robin across class composition, "
            "object density, and median box scale; negatives use a stable shuffle."
        ),
        "splits": {
            split: {
                "total": _record_summary(selected_totals[split]),
                "sources": manifest_sources[split],
            }
            for split in SPLITS
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Build and report a reproducible balanced detector dataset."""
    arguments = build_parser().parse_args()
    quotas = _load_quotas(arguments.quota_config)
    manifest = prepare(
        source_root=arguments.source,
        output=arguments.output,
        combined_input=arguments.combined_input,
        combined_output=arguments.combined_output,
        quotas=quotas,
        seed=arguments.seed,
        replace=arguments.replace,
    )
    print(
        json.dumps(
            {split: values["total"] for split, values in manifest["splits"].items()},
            indent=2,
        )
    )
    print(f"Balanced source: {arguments.output.resolve()}")
    print(f"Balanced combined YAML: {arguments.combined_output.resolve()}")
    print("Training was not started.")


if __name__ == "__main__":
    main()
