"""Download and normalize additional public Roboflow fastener datasets."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
OUTPUT_SPLITS = ("train", "valid", "test")
TARGET_IDS = {"bolt": 0, "nut": 1}
REVIEW_FIELDS = (
    "source",
    "split",
    "image",
    "bolt_boxes",
    "nut_boxes",
    "ignored_boxes",
    "review_status",
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """
    Describe one versioned Roboflow source and its strict class mapping.

    Parameters
    ----------
    key : str
        Stable local identifier used in cache and output filenames.
    workspace : str
        Roboflow workspace slug.
    project : str
        Roboflow project slug.
    version : int
        Immutable dataset version exported by the importer.
    title : str
        Human-readable dataset title.
    url : str
        Public project page used for provenance.
    license_name : str
        License reported by the publisher.
    target_aliases : dict[str, str]
        Normalized source-class names mapped to ``bolt`` or ``nut``.
    """

    key: str
    workspace: str
    project: str
    version: int
    title: str
    url: str
    license_name: str
    target_aliases: dict[str, str]


SOURCES = (
    SourceSpec(
        key="fastener_object_detection",
        workspace="automated-disassembly",
        project="fastener-object-detection",
        version=1,
        title="Fastener Object Detection",
        url=(
            "https://universe.roboflow.com/automated-disassembly/"
            "fastener-object-detection/dataset/1"
        ),
        license_name="CC BY 4.0",
        target_aliases={"hexbolt": "bolt", "hexnut": "nut"},
    ),
    SourceSpec(
        key="screw_nut_bolt",
        workspace="logesh-s-workspace",
        project="screw-nut-bolt",
        version=1,
        title="screw-nut-bolt",
        url=(
            "https://universe.roboflow.com/logesh-s-workspace/"
            "screw-nut-bolt/dataset/1"
        ),
        license_name="CC BY 4.0",
        target_aliases={"bolt": "bolt", "nut": "nut"},
    ),
    SourceSpec(
        key="stresstest_tool_tray",
        workspace="tool-tray-yolo-training-dataset",
        project="stresstest-troll-needle-nut-bolt-and-scissor",
        version=3,
        title="Stresstest @Troll - Needle, nut, bolt and scissor",
        url=(
            "https://universe.roboflow.com/tool-tray-yolo-training-dataset/"
            "stresstest-troll-needle-nut-bolt-and-scissor"
        ),
        license_name="CC BY 4.0",
        target_aliases={"bolt": "bolt", "nut": "nut"},
    ),
)


@dataclass(frozen=True, slots=True)
class SourceImage:
    """
    Store the source paths and split assignment for one exported image.

    Parameters
    ----------
    image_path : Path
        Exported image file.
    label_path : Path
        Matching YOLO annotation file.
    declared_split : str
        Split declared by the publisher.
    output_split : str
        Leakage-safe split selected by the importer.
    family : str
        Filename-derived Roboflow source-family identifier.
    relative_path : Path
        Image path relative to its declared image directory.
    """

    image_path: Path
    label_path: Path
    declared_split: str
    output_split: str
    family: str
    relative_path: Path


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line arguments for secure downloading and conversion.

    Returns
    -------
    argparse.ArgumentParser
        Parser with repository-local cache and output defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download three public Roboflow datasets, retain complete bolt/nut "
            "boxes, and add the normalized source to data/combined.yaml."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/external_sources/roboflow_additions"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/roboflow_additions")
    )
    parser.add_argument("--combined", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/external_benchmark/images")
    )
    parser.add_argument("--api-key-environment", default="ROBOFLOW_API_KEY")
    parser.add_argument(
        "--source",
        choices=tuple(source.key for source in SOURCES),
        action="append",
        help="Import only selected sources; omit to import all three.",
    )
    parser.add_argument(
        "--max-variants-per-family",
        type=int,
        default=1,
        help="Maximum retained files per Roboflow pre-augmentation image family.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--skip-combined-update", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate the SHA-256 digest of one file.

    Parameters
    ----------
    path : Path
        File to hash.

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


def _normalized_class_name(value: str) -> str:
    """
    Normalize punctuation and case in a source class name.

    Parameters
    ----------
    value : str
        Class name declared by a source dataset.

    Returns
    -------
    str
        Lowercase alphanumeric identifier.
    """
    return "".join(character for character in value.casefold() if character.isalnum())


def _export_link(payload: Any) -> str | None:
    """
    Find the first HTTP export link in a nested Roboflow response.

    Parameters
    ----------
    payload : object
        Parsed Roboflow API response.

    Returns
    -------
    str or None
        Temporary archive URL when present.
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


def _download_source(source: SourceSpec, destination: Path, api_key: str) -> None:
    """
    Download one official YOLOv8 export without persisting the API key.

    Parameters
    ----------
    source : SourceSpec
        Versioned dataset to export.
    destination : Path
        Local ZIP cache path.
    api_key : str
        Roboflow API key held only in process memory.
    """
    endpoint = (
        f"https://api.roboflow.com/{source.workspace}/{source.project}/"
        f"{source.version}/yolov8?api_key="
        + urllib.parse.quote(api_key, safe="")
    )
    request = urllib.request.Request(endpoint, headers={"User-Agent": "nut-bolt-dataset-importer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"Roboflow export request failed for {source.key}: HTTP {error.code}"
        ) from None
    link = _export_link(payload.get("export", payload))
    if not link:
        raise ValueError(f"Roboflow returned no export link for {source.key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        try:
            with urllib.request.urlopen(link, timeout=300) as response, temporary.open("wb") as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
        except urllib.error.HTTPError as error:
            raise ValueError(
                f"Roboflow archive download failed for {source.key}: HTTP {error.code}"
            ) from None
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Downloaded archive is corrupt: {source.key}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_extract(archive_path: Path, destination: Path, replace: bool) -> Path:
    """
    Safely extract an archive and return its unique dataset YAML path.

    Parameters
    ----------
    archive_path : Path
        Cached Roboflow ZIP export.
    destination : Path
        Extraction directory.
    replace : bool
        Whether an existing extraction should be rebuilt.

    Returns
    -------
    Path
        Located ``data.yaml`` file.
    """
    if destination.exists() and replace:
        shutil.rmtree(destination)
    if not destination.exists():
        destination.mkdir(parents=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe ZIP member in {archive_path}: {member.filename}")
            archive.extractall(destination)
    yaml_paths = sorted(destination.rglob("data.yaml"))
    if len(yaml_paths) != 1:
        raise ValueError(
            f"Expected exactly one data.yaml below {destination}, found {len(yaml_paths)}"
        )
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
        Source class names indexed by numeric identifier.
    """
    names = document.get("names")
    if isinstance(names, list):
        return {index: str(name).strip() for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name).strip() for index, name in names.items()}
    raise ValueError("Source data.yaml has no valid names declaration")


def _dataset_root(data_path: Path, document: dict[str, Any]) -> Path:
    """
    Resolve the root declared by a YOLO dataset configuration.

    Parameters
    ----------
    data_path : Path
        Source YAML path.
    document : dict
        Parsed YAML payload.

    Returns
    -------
    Path
        Absolute dataset root.
    """
    root = data_path.parent
    declared = document.get("path")
    if declared:
        path = Path(str(declared))
        root = path if path.is_absolute() else root / path
    return root.resolve()


def _declared_image_roots(
    data_path: Path, document: dict[str, Any], split: str
) -> list[Path]:
    """
    Resolve existing image directories declared for one source split.

    Parameters
    ----------
    data_path : Path
        Source dataset YAML path.
    document : dict
        Parsed YAML payload.
    split : str
        Canonical output split name.

    Returns
    -------
    list[Path]
        Existing image directories for the requested split.
    """
    source_key = "val" if split == "valid" else split
    declared = document.get(source_key)
    if declared is None and split == "valid":
        declared = document.get("valid")
    values = declared if isinstance(declared, list) else [declared]
    root = _dataset_root(data_path, document)
    roots: list[Path] = []
    for value in values:
        if value is None:
            continue
        relative = Path(str(value))
        candidates = (
            relative if relative.is_absolute() else root / relative,
            data_path.parent / split / "images",
        )
        selected = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if selected is not None and selected not in roots:
            roots.append(selected)
    return roots


def _source_family(path: Path) -> str:
    """
    Derive the original-image family from a Roboflow export filename.

    Parameters
    ----------
    path : Path
        Exported image path.

    Returns
    -------
    str
        Case-insensitive family identifier without ``.rf.<hash>``.
    """
    return path.name.split(".rf.", 1)[0].casefold()


def _hashed_split(source_key: str, family: str) -> str:
    """
    Assign a deterministic 80/10/10 split to an unsplit source family.

    Parameters
    ----------
    source_key : str
        Stable dataset identifier.
    family : str
        Original-image family identifier.

    Returns
    -------
    str
        ``train``, ``valid``, or ``test``.
    """
    value = int(hashlib.sha256(f"{source_key}:{family}".encode()).hexdigest()[:8], 16) % 1000
    if value < 800:
        return "train"
    if value < 900:
        return "valid"
    return "test"


def _collect_source_images(
    source: SourceSpec, data_path: Path, document: dict[str, Any]
) -> list[SourceImage]:
    """
    Enumerate images and choose leakage-safe output splits.

    Parameters
    ----------
    source : SourceSpec
        Dataset being converted.
    data_path : Path
        Extracted source YAML path.
    document : dict
        Parsed source YAML payload.

    Returns
    -------
    list[SourceImage]
        Source records with deterministic family-level split assignments.
    """
    raw: list[tuple[Path, Path, str, str, Path]] = []
    declared_nonempty: set[str] = set()
    for split in OUTPUT_SPLITS:
        for image_root in _declared_image_roots(data_path, document, split):
            images = sorted(
                path
                for path in image_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if images:
                declared_nonempty.add(split)
            label_root = image_root.parent / "labels"
            for image_path in images:
                relative = image_path.relative_to(image_root)
                raw.append(
                    (
                        image_path,
                        (label_root / relative).with_suffix(".txt"),
                        split,
                        _source_family(image_path),
                        relative,
                    )
                )
    if not raw:
        raise ValueError(f"No images were found for {source.key}")

    train_only = declared_nonempty == {"train"}
    family_splits: dict[str, str] = {}
    for _, _, declared_split, family, _ in raw:
        desired = _hashed_split(source.key, family) if train_only else declared_split
        previous = family_splits.get(family)
        if previous is None:
            family_splits[family] = desired
        elif previous != desired:
            family_splits[family] = _hashed_split(source.key, family)

    return [
        SourceImage(
            image_path=image_path,
            label_path=label_path,
            declared_split=declared_split,
            output_split=family_splits[family],
            family=family,
            relative_path=relative,
        )
        for image_path, label_path, declared_split, family, relative in raw
    ]


def _convert_label(
    label_path: Path,
    names: dict[int, str],
    aliases: dict[str, str],
) -> tuple[list[str], Counter[str], int]:
    """
    Retain target boxes and count annotations ignored as background.

    Parameters
    ----------
    label_path : Path
        Source YOLO label file; a missing file is treated as a negative image.
    names : dict[int, str]
        Source class names keyed by numeric identifier.
    aliases : dict[str, str]
        Normalized class names mapped to project target classes.

    Returns
    -------
    tuple[list[str], Counter[str], int]
        Converted lines, source-class counts, and ignored box count.
    """
    if not label_path.is_file():
        return [], Counter(), 0
    output: list[str] = []
    source_counts: Counter[str] = Counter()
    ignored = 0
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Malformed annotation at {label_path}:{line_number}")
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid numeric value at {label_path}:{line_number}") from error
        if class_id not in names:
            raise ValueError(f"Unknown class id at {label_path}:{line_number}")
        if any(not 0.0 <= coordinate <= 1.0 for coordinate in coordinates):
            raise ValueError(f"Coordinate outside [0, 1] at {label_path}:{line_number}")

        source_name = names[class_id]
        source_counts[source_name] += 1
        target_name = aliases.get(_normalized_class_name(source_name))
        if target_name is None:
            ignored += 1
            continue
        target_id = str(TARGET_IDS[target_name])
        if len(coordinates) == 4:
            output.append(" ".join([target_id, *values[1:]]))
            continue
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"Malformed polygon at {label_path}:{line_number}")
        x_values = coordinates[0::2]
        y_values = coordinates[1::2]
        x1, x2 = min(x_values), max(x_values)
        y1, y2 = min(y_values), max(y_values)
        output.append(
            f"{target_id} {(x1 + x2) / 2.0:.8f} {(y1 + y2) / 2.0:.8f} "
            f"{x2 - x1:.8f} {y2 - y1:.8f}"
        )
    return output, source_counts, ignored


def _existing_hashes(
    combined_path: Path,
    benchmark_root: Path,
    excluded_root: Path | None = None,
) -> tuple[set[str], set[str]]:
    """
    Hash current training and frozen benchmark images for exact deduplication.

    Parameters
    ----------
    combined_path : Path
        Existing combined YOLO configuration.
    benchmark_root : Path
        Frozen benchmark image directory.
    excluded_root : Path, optional
        Generated output being rebuilt; its previous files must not deduplicate
        their own replacements.

    Returns
    -------
    tuple[set[str], set[str]]
        Current-training hashes and benchmark hashes.
    """
    training: set[str] = set()
    if combined_path.is_file():
        document: dict[str, Any] = yaml.safe_load(combined_path.read_text(encoding="utf-8")) or {}
        root = combined_path.parent
        if document.get("path"):
            declared_root = Path(str(document["path"]))
            root = declared_root if declared_root.is_absolute() else root / declared_root
        for key in ("train", "val", "valid", "test"):
            declared = document.get(key)
            values = declared if isinstance(declared, list) else [declared]
            for value in values:
                if value is None:
                    continue
                path = Path(str(value))
                image_root = (path if path.is_absolute() else root / path).resolve()
                excluded = excluded_root.resolve() if excluded_root is not None else None
                if excluded is not None and (
                    image_root == excluded or excluded in image_root.parents
                ):
                    continue
                if image_root.is_dir():
                    training.update(
                        _sha256(image_path)
                        for image_path in image_root.rglob("*")
                        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
                    )
    benchmark = (
        {
            _sha256(image_path)
            for image_path in benchmark_root.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
        }
        if benchmark_root.is_dir()
        else set()
    )
    return training, benchmark


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create a fresh normalized YOLO output directory.

    Parameters
    ----------
    output : Path
        Generated dataset root.
    replace : bool
        Whether an existing generated dataset may be removed.
    """
    if output.exists():
        if not replace:
            raise ValueError(f"Output exists: {output}; pass --replace to rebuild it")
        shutil.rmtree(output)
    for split in OUTPUT_SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)


def _link_or_copy(source: Path, destination: Path) -> None:
    """
    Hard-link an image when possible and otherwise copy it.

    Parameters
    ----------
    source : Path
        Extracted source image.
    destination : Path
        Normalized dataset image path.
    """
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _convert_sources(
    selected: tuple[SourceSpec, ...],
    data_paths: dict[str, Path],
    output: Path,
    combined: Path,
    benchmark: Path,
    max_variants_per_family: int,
    replace: bool,
) -> dict[str, Any]:
    """
    Convert selected sources into one strict, deduplicated two-class dataset.

    Parameters
    ----------
    selected : tuple[SourceSpec, ...]
        Sources requested by the user.
    data_paths : dict[str, Path]
        Extracted YAML path keyed by source identifier.
    output : Path
        Generated normalized dataset root.
    combined : Path
        Existing training configuration used for deduplication.
    benchmark : Path
        Frozen benchmark image directory excluded from training.
    max_variants_per_family : int
        Maximum retained images per Roboflow source family.
    replace : bool
        Whether an existing generated dataset may be rebuilt.

    Returns
    -------
    dict
        Reproducible dataset manifest.
    """
    if max_variants_per_family < 1:
        raise ValueError("max_variants_per_family must be at least 1")
    training_hashes, benchmark_hashes = _existing_hashes(
        combined, benchmark, excluded_root=output
    )
    _prepare_output(output, replace)
    seen_hashes: set[str] = set()
    split_totals = {split: Counter() for split in OUTPUT_SPLITS}
    source_manifests: list[dict[str, Any]] = []
    review_rows: list[dict[str, object]] = []

    for source in selected:
        data_path = data_paths[source.key]
        document: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        names = _class_names(document)
        normalized_names = {_normalized_class_name(name) for name in names.values()}
        missing_targets = set(source.target_aliases) - normalized_names
        available_aliases = set(source.target_aliases) & normalized_names
        if not available_aliases:
            raise ValueError(
                f"{source.key} contains none of its expected target classes; "
                f"found {list(names.values())}"
            )
        if missing_targets:
            print(
                f"Warning: {source.key} export does not contain optional classes "
                f"{sorted(missing_targets)}; available target aliases: "
                f"{sorted(available_aliases)}"
            )

        records = _collect_source_images(source, data_path, document)
        family_counts: Counter[tuple[str, str]] = Counter()
        source_counts: Counter[str] = Counter()
        source_splits = {split: Counter() for split in OUTPUT_SPLITS}
        removed: Counter[str] = Counter()
        missing_label_files = 0

        for record in records:
            family_key = (record.output_split, record.family)
            if family_counts[family_key] >= max_variants_per_family:
                removed["offline_variants"] += 1
                continue
            digest = _sha256(record.image_path)
            if digest in benchmark_hashes:
                removed["external_benchmark"] += 1
                continue
            if digest in training_hashes:
                removed["existing_training_data"] += 1
                continue
            if digest in seen_hashes:
                removed["within_new_sources"] += 1
                continue

            lines, label_counts, ignored = _convert_label(
                record.label_path, names, source.target_aliases
            )
            missing_label_files += int(not record.label_path.is_file())
            source_counts.update(label_counts)
            family_counts[family_key] += 1
            seen_hashes.add(digest)

            relative_digest = hashlib.sha256(
                record.relative_path.as_posix().encode("utf-8")
            ).hexdigest()[:10]
            stem = f"{source.key}__{record.image_path.stem}__{relative_digest}"
            target_image = output / record.output_split / "images" / f"{stem}{record.image_path.suffix.lower()}"
            target_label = output / record.output_split / "labels" / f"{stem}.txt"
            _link_or_copy(record.image_path, target_image)
            target_label.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

            bolt_boxes = sum(line.startswith("0 ") for line in lines)
            nut_boxes = sum(line.startswith("1 ") for line in lines)
            counters = source_splits[record.output_split]
            counters["images"] += 1
            counters["bolt_boxes"] += bolt_boxes
            counters["nut_boxes"] += nut_boxes
            counters["negative_images"] += int(not lines)
            counters["ignored_boxes"] += ignored
            split_totals[record.output_split]["images"] += 1
            split_totals[record.output_split]["bolt_boxes"] += bolt_boxes
            split_totals[record.output_split]["nut_boxes"] += nut_boxes
            split_totals[record.output_split]["negative_images"] += int(not lines)
            split_totals[record.output_split]["ignored_boxes"] += ignored
            review_rows.append(
                {
                    "source": source.key,
                    "split": record.output_split,
                    "image": target_image.name,
                    "bolt_boxes": bolt_boxes,
                    "nut_boxes": nut_boxes,
                    "ignored_boxes": ignored,
                    "review_status": "source_annotation_pending_spot_check",
                }
            )

        source_manifests.append(
            {
                "key": source.key,
                "title": source.title,
                "url": source.url,
                "workspace": source.workspace,
                "project": source.project,
                "version": source.version,
                "license": source.license_name,
                "class_mapping": source.target_aliases,
                "missing_declared_target_aliases": sorted(missing_targets),
                "source_classes": names,
                "source_boxes": dict(source_counts),
                "splits": {split: dict(counts) for split, counts in source_splits.items()},
                "removed": dict(removed),
                "missing_label_files_treated_as_negative": missing_label_files,
                "archive_sha256": _sha256(data_path.parent.parent / f"{source.key}.zip")
                if (data_path.parent.parent / f"{source.key}.zip").is_file()
                else None,
            }
        )

    if not review_rows:
        raise ValueError("All requested images were removed; normalized dataset would be empty")
    if not all(
        sum(split_totals[split][f"{class_name}_boxes"] for split in OUTPUT_SPLITS) > 0
        for class_name in TARGET_IDS
    ):
        raise ValueError("Normalized sources must contain at least one bolt and one nut box")

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
        writer.writerows(review_rows)
    manifest = {
        "description": "Strict bolt/nut conversion of three public Roboflow sources.",
        "target_classes": ["bolt", "nut"],
        "ignored_classes": "Retained images, omitted boxes; treated as background.",
        "deduplication": (
            "SHA-256 against current training data, frozen benchmark, and new sources; "
            "Roboflow filename-family cap prevents offline-augmentation multiplication."
        ),
        "max_variants_per_family": max_variants_per_family,
        "splits": {split: dict(counts) for split, counts in split_totals.items()},
        "sources": source_manifests,
        "review_worksheet": "review.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _update_combined(combined_path: Path, output: Path) -> None:
    """
    Append the normalized source to every combined dataset split idempotently.

    Parameters
    ----------
    combined_path : Path
        Combined YOLO YAML to update.
    output : Path
        Newly generated strict dataset root.
    """
    if not combined_path.is_file():
        raise ValueError(f"Combined dataset YAML was not found: {combined_path}")
    document: dict[str, Any] = yaml.safe_load(combined_path.read_text(encoding="utf-8")) or {}
    for yaml_key, directory in (("train", "train"), ("val", "valid"), ("test", "test")):
        relative = Path(
            os.path.relpath(
                (output / directory / "images").resolve(),
                start=combined_path.parent.resolve(),
            )
        ).as_posix()
        current = document.get(yaml_key, [])
        values = current if isinstance(current, list) else [current]
        values = [str(value) for value in values if value is not None]
        if relative not in values:
            positive_end = next(
                (
                    index
                    for index, value in enumerate(values)
                    if any(
                        marker in value
                        for marker in ("negatives/", "hard_negatives/", "manual_hard_negatives/")
                    )
                ),
                len(values),
            )
            values.insert(positive_end, relative)
        document[yaml_key] = values
    document["names"] = ["bolt", "nut"]
    combined_path.write_text(
        "# Generated dataset composition. Training sources only; benchmark is excluded.\n"
        + yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _selected_sources(keys: list[str] | None) -> tuple[SourceSpec, ...]:
    """
    Resolve optional source keys while preserving canonical order.

    Parameters
    ----------
    keys : list[str] or None
        Repeated ``--source`` values.

    Returns
    -------
    tuple[SourceSpec, ...]
        Selected immutable source specifications.
    """
    selected = set(keys or (source.key for source in SOURCES))
    return tuple(source for source in SOURCES if source.key in selected)


def main() -> None:
    """Securely download, normalize, deduplicate, and register all selected sources."""
    arguments = build_parser().parse_args()
    selected = _selected_sources(arguments.source)
    if not selected:
        raise ValueError("Select at least one source")
    arguments.cache_root.mkdir(parents=True, exist_ok=True)

    archives = {
        source.key: arguments.cache_root / f"{source.key}.zip" for source in selected
    }
    missing = [
        source
        for source in selected
        if arguments.redownload or not archives[source.key].is_file()
    ]
    api_key: str | None = None
    if missing:
        api_key = os.getenv(arguments.api_key_environment)
        if not api_key:
            api_key = getpass.getpass(
                "Roboflow API key (hidden, used only by this process): "
            ).strip()
        if not api_key:
            raise ValueError("A Roboflow API key is required for official exports")
        for source in missing:
            print(f"Downloading {source.title} v{source.version} ...")
            _download_source(source, archives[source.key], api_key)
        api_key = None

    data_paths: dict[str, Path] = {}
    for source in selected:
        extract_root = arguments.cache_root / source.key
        data_paths[source.key] = _safe_extract(
            archives[source.key], extract_root, arguments.replace
        )
    manifest = _convert_sources(
        selected=selected,
        data_paths=data_paths,
        output=arguments.output,
        combined=arguments.combined,
        benchmark=arguments.benchmark,
        max_variants_per_family=arguments.max_variants_per_family,
        replace=arguments.replace,
    )
    if not arguments.skip_combined_update:
        _update_combined(arguments.combined, arguments.output)
    print(json.dumps(manifest["splits"], indent=2))
    print(f"Normalized dataset: {arguments.output.resolve()}")
    print(f"Review worksheet: {(arguments.output / 'review.csv').resolve()}")
    if not arguments.skip_combined_update:
        print(f"Updated combined YAML: {arguments.combined.resolve()}")
    print("Training was not started.")


if __name__ == "__main__":
    main()
