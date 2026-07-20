"""Download and convert ARG_FIXINGS2 into a strict bolt/nut training source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
OUTPUT_SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
TARGET_IDS = {"bolt": 0, "nut": 1}
REVIEW_FIELDS = ("split", "image", "bolt_boxes", "nut_boxes", "review_status")


def build_parser() -> argparse.ArgumentParser:
    """
    Create arguments for authenticated download and strict conversion.

    Returns
    -------
    argparse.ArgumentParser
        Parser with repository-local cache and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the CC BY 4.0 ARG_FIXINGS2 training dataset."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/external_sources/arg_fixings2_export.zip"),
    )
    parser.add_argument(
        "--extract",
        type=Path,
        default=Path("data/external_sources/arg_fixings2_export"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/arg_fixings_training"))
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/external_benchmark/images")
    )
    parser.add_argument(
        "--max-train-variants-per-source",
        type=int,
        default=1,
        help=(
            "Maximum Roboflow offline-augmented variants retained for each "
            "original training image (default: 1)."
        ),
    )
    parser.add_argument("--api-key-environment", default="ROBOFLOW_API_KEY")
    parser.add_argument("--replace", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate the SHA-256 digest for one file.

    Parameters
    ----------
    path : Path
        File whose bytes are hashed.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_link(payload: Any) -> str | None:
    """
    Find the first HTTP export link in a Roboflow API response.

    Parameters
    ----------
    payload : object
        Parsed JSON response containing nested export metadata.

    Returns
    -------
    str or None
        Download URL when one is present.
    """
    if isinstance(payload, str) and payload.startswith(("http://", "https://")):
        return payload
    if isinstance(payload, dict):
        preferred = payload.get("link")
        if isinstance(preferred, str) and preferred.startswith(("http://", "https://")):
            return preferred
        for value in payload.values():
            found = _export_link(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _export_link(value)
            if found:
                return found
    return None


def _download(cache: Path, api_key: str) -> None:
    """
    Request and download the official Roboflow YOLOv8 export.

    Parameters
    ----------
    cache : Path
        Local ZIP cache destination.
    api_key : str
        Roboflow API key read only from the selected environment variable.
    """
    endpoint = (
        "https://api.roboflow.com/bolts/arg_fixings2/1/yolov8?api_key="
        + urllib.parse.quote(api_key, safe="")
    )
    with urllib.request.urlopen(endpoint, timeout=60) as response:
        payload = json.load(response)
    link = _export_link(payload.get("export", payload))
    if not link:
        raise ValueError("Roboflow response did not contain an export download link")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(link, timeout=120) as response, cache.open("wb") as file:
        shutil.copyfileobj(response, file, length=1024 * 1024)


def _extract(cache: Path, destination: Path, replace: bool) -> Path:
    """
    Extract a cached dataset and locate its YAML configuration.

    Parameters
    ----------
    cache : Path
        Downloaded ZIP archive.
    destination : Path
        Extraction directory.
    replace : bool
        Whether an existing extraction may be rebuilt.

    Returns
    -------
    Path
        Located source dataset YAML.
    """
    if destination.exists() and replace:
        shutil.rmtree(destination)
    if not destination.exists():
        destination.mkdir(parents=True)
        with zipfile.ZipFile(cache) as archive:
            archive.extractall(destination)
    yaml_paths = sorted(destination.rglob("data.yaml"))
    if not yaml_paths:
        raise ValueError(f"No data.yaml was found below {destination}")
    return yaml_paths[0]


def _class_names(document: dict[str, Any]) -> dict[int, str]:
    """
    Normalize list- or mapping-style YOLO class declarations.

    Parameters
    ----------
    document : dict
        Parsed source dataset YAML.

    Returns
    -------
    dict[int, str]
        Lowercase class names indexed by source class identifier.
    """
    names = document.get("names")
    if isinstance(names, list):
        return {index: str(name).strip().lower() for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name).strip().lower() for index, name in names.items()}
    raise ValueError("Source data.yaml has no valid names declaration")


def _declared_root(data_path: Path, document: dict[str, Any]) -> Path:
    """
    Resolve the dataset root declared by a YOLO YAML file.

    Parameters
    ----------
    data_path : Path
        Source YAML path.
    document : dict
        Parsed YAML document.

    Returns
    -------
    Path
        Absolute source dataset root.
    """
    root = data_path.parent
    declared = document.get("path")
    if declared:
        path = Path(str(declared))
        root = path if path.is_absolute() else root / path
    return root.resolve()


def _convert_label(
    source: Path, names: dict[int, str]
) -> tuple[list[str], Counter[str]]:
    """
    Retain complete bolt/nut boxes and ignore screw/washer boxes.

    Parameters
    ----------
    source : Path
        Source YOLO label file.
    names : dict[int, str]
        Source class mapping.

    Returns
    -------
    tuple[list[str], collections.Counter]
        Converted label lines and source-class counts.
    """
    output: list[str] = []
    counts: Counter[str] = Counter()
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Malformed label at {source}:{line_number}")
        class_id = int(values[0])
        if class_id not in names:
            raise ValueError(f"Unknown class at {source}:{line_number}")
        class_name = names[class_id]
        counts[class_name] += 1
        coordinates = [float(value) for value in values[1:]]
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"Coordinates outside [0, 1] at {source}:{line_number}")
        if class_name in TARGET_IDS:
            target_id = str(TARGET_IDS[class_name])
            if len(coordinates) == 4:
                output.append(" ".join([target_id, *values[1:]]))
                continue
            if len(coordinates) < 6 or len(coordinates) % 2:
                raise ValueError(f"Malformed polygon at {source}:{line_number}")
            x_values = coordinates[0::2]
            y_values = coordinates[1::2]
            x1, x2 = min(x_values), max(x_values)
            y1, y2 = min(y_values), max(y_values)
            output.append(
                f"{target_id} {(x1 + x2) / 2.0:.8f} {(y1 + y2) / 2.0:.8f} "
                f"{x2 - x1:.8f} {y2 - y1:.8f}"
            )
    return output, counts


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create empty standard YOLO split directories.

    Parameters
    ----------
    output : Path
        Generated training source root.
    replace : bool
        Whether an existing output may be removed.
    """
    if output.exists():
        if not replace:
            raise ValueError(f"Output exists: {output}; pass --replace")
        shutil.rmtree(output)
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)


def _benchmark_hashes(root: Path) -> set[str]:
    """
    Hash the frozen benchmark to enforce source separation.

    Parameters
    ----------
    root : Path
        Benchmark image directory.

    Returns
    -------
    set[str]
        Image byte digests, or an empty set when the benchmark is unavailable.
    """
    if not root.is_dir():
        return set()
    return {
        _sha256(path)
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _convert(
    data_path: Path,
    output: Path,
    benchmark: Path,
    replace: bool,
    max_train_variants_per_source: int,
) -> dict[str, Any]:
    """
    Convert all source splits and write provenance metadata.

    Parameters
    ----------
    data_path : Path
        Extracted Roboflow dataset YAML.
    output : Path
        Strict two-class training source destination.
    benchmark : Path
        Frozen benchmark images excluded by byte digest.
    replace : bool
        Whether generated output may be rebuilt.
    max_train_variants_per_source : int
        Maximum offline-augmented variants kept per original train image.

    Returns
    -------
    dict
        Split counts, source counts, and exclusion evidence.
    """
    if max_train_variants_per_source < 1:
        raise ValueError("max_train_variants_per_source must be at least 1")
    document: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = _class_names(document)
    root = _declared_root(data_path, document)
    benchmark_digests = _benchmark_hashes(benchmark)
    _prepare_output(output, replace)
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "valid", "test")
    }
    source_counts: Counter[str] = Counter()
    seen: set[str] = set()
    review: list[dict[str, object]] = []
    skipped_benchmark = 0
    skipped_duplicate = 0
    skipped_offline_variant = 0
    retained_train_families: Counter[str] = Counter()

    for declared_split, output_split in OUTPUT_SPLITS.items():
        declared = document.get(declared_split)
        if declared is None:
            continue
        image_root = Path(str(declared))
        if image_root.is_absolute():
            candidates = (image_root,)
        else:
            # Some Roboflow ZIP exports declare paths such as
            # ``../train/images`` while placing data.yaml next to ``train``.
            # Prefer the declared path, then support that known export layout.
            candidates = (
                root / image_root,
                data_path.parent / output_split / "images",
            )
        image_root = next(
            (candidate.resolve() for candidate in candidates if candidate.is_dir()),
            candidates[0].resolve(),
        )
        if not image_root.is_dir():
            continue
        label_root = image_root.parent / "labels"
        for image_path in sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            family = image_path.name.split(".rf.", 1)[0].casefold()
            if (
                output_split == "train"
                and retained_train_families[family] >= max_train_variants_per_source
            ):
                skipped_offline_variant += 1
                continue
            digest = _sha256(image_path)
            if digest in benchmark_digests:
                skipped_benchmark += 1
                continue
            if digest in seen:
                skipped_duplicate += 1
                continue
            seen.add(digest)
            if output_split == "train":
                retained_train_families[family] += 1
            relative = image_path.relative_to(image_root)
            label_path = (label_root / relative).with_suffix(".txt")
            if not label_path.is_file():
                raise ValueError(f"Missing source label: {label_path}")
            lines, counts = _convert_label(label_path, names)
            source_counts.update(counts)
            target = f"arg_{image_path.name}"
            shutil.copy2(image_path, output / output_split / "images" / target)
            (output / output_split / "labels" / f"{Path(target).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            split_counts[output_split]["images"] += 1
            split_counts[output_split]["bolt_boxes"] += sum(
                line.startswith("0 ") for line in lines
            )
            split_counts[output_split]["nut_boxes"] += sum(
                line.startswith("1 ") for line in lines
            )
            split_counts[output_split]["negative_images"] += int(not lines)
            review.append(
                {
                    "split": output_split,
                    "image": target,
                    "bolt_boxes": sum(line.startswith("0 ") for line in lines),
                    "nut_boxes": sum(line.startswith("1 ") for line in lines),
                    "review_status": "source_annotation_pending_spot_check",
                }
            )

    if not review:
        raise ValueError(
            "No images were found in the dataset splits declared by "
            f"{data_path}. Check the YAML split paths and extracted directory layout."
        )

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
    with (output / "review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)
    manifest = {
        "source": {
            "title": "ARG_FIXINGS2",
            "url": "https://universe.roboflow.com/bolts/arg_fixings2",
            "version": 1,
            "license": "CC BY 4.0",
            "mapping": "Bolt -> bolt; Nut -> nut; Screw and Washer -> background",
            "annotation_note": "Source reports manual annotation with model/SAM assistance.",
        },
        "archive_sha256": _sha256(data_path.parents[1] / "arg_fixings2_export.zip")
        if (data_path.parents[1] / "arg_fixings2_export.zip").is_file()
        else None,
        "splits": {name: dict(counts) for name, counts in split_counts.items()},
        "source_boxes": dict(source_counts),
        "duplicates_removed": {
            "against_external_benchmark": skipped_benchmark,
            "within_source": skipped_duplicate,
            "offline_augmented_variants": skipped_offline_variant,
        },
        "max_train_variants_per_source": max_train_variants_per_source,
        "review_worksheet": "review.csv",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Download when needed, convert the source, and print its manifest."""
    arguments = build_parser().parse_args()
    if not arguments.cache.is_file():
        api_key = os.getenv(arguments.api_key_environment)
        if not api_key:
            raise ValueError(
                f"Set {arguments.api_key_environment} in the current shell; "
                "do not put the API key in source code"
            )
        _download(arguments.cache, api_key)
    data_path = _extract(arguments.cache, arguments.extract, arguments.replace)
    manifest = _convert(
        data_path,
        arguments.output,
        arguments.benchmark,
        arguments.replace,
        arguments.max_train_variants_per_source,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Strict training source: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
