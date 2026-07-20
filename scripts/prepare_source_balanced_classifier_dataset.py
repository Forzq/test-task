"""Build a deduplicated source-balanced crop-classifier dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("train", "val", "test")
CLASSES = ("bolt", "nut", "other")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GENERATED_MARKER = ".generated-source-balanced-classifier"


@dataclass(frozen=True, slots=True)
class CropRecord:
    """
    Store one validated source crop.

    Parameters
    ----------
    path : Path
        Absolute source image path.
    source : str
        Stable source identifier.
    split : str
        Classification split name.
    class_name : str
        Crop target class.
    digest : str
        SHA-256 image digest.
    """

    path: Path
    source: str
    split: str
    class_name: str
    digest: str


@dataclass(frozen=True, slots=True)
class BalanceConfig:
    """
    Configure deterministic source-balanced crop selection.

    Parameters
    ----------
    legacy : Path
        Legacy classifier dataset root.
    dataset3 : Path
        Dataset 3 classifier dataset root.
    output : Path
        Generated balanced dataset root.
    train_per_source_class : int
        Train crops retained from each source and class.
    val_per_source_class : int
        Validation crops retained from each source and class.
    test_per_source_class : int
        Test crops retained from each source and class.
    seed : int
        Deterministic sampling seed.
    replace : bool
        Whether a marked generated output may be replaced.
    """

    legacy: Path
    dataset3: Path
    output: Path
    train_per_source_class: int
    val_per_source_class: int
    test_per_source_class: int
    seed: int
    replace: bool


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for balanced crop preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with project-local defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mix legacy and Dataset 3 classifier crops with equal source quotas "
            "and exact-deduplication."
        )
    )
    parser.add_argument("--legacy", type=Path, default=Path("data/classifier"))
    parser.add_argument(
        "--dataset3", type=Path, default=Path("data/classifier_dataset3_v9")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/classifier_source_balanced_v10")
    )
    parser.add_argument("--train-per-source-class", type=int, default=4000)
    parser.add_argument("--val-per-source-class", type=int, default=999)
    parser.add_argument("--test-per-source-class", type=int, default=572)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate a file SHA-256 digest.

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


def _validate_root(root: Path) -> None:
    """
    Validate the expected classification folder layout.

    Parameters
    ----------
    root : Path
        Classification dataset root.

    Raises
    ------
    ValueError
        Raised when a required split or class directory is missing.
    """
    for split in SPLITS:
        for class_name in CLASSES:
            directory = root / split / class_name
            if not directory.is_dir():
                raise ValueError(f"Classifier directory was not found: {directory}")


def _records(root: Path, source: str) -> list[CropRecord]:
    """
    Read and validate every crop below one source root.

    Parameters
    ----------
    root : Path
        Classification dataset root.
    source : str
        Source identifier written to output provenance.

    Returns
    -------
    list[CropRecord]
        Validated crop records with content hashes.
    """
    _validate_root(root)
    records: list[CropRecord] = []
    for split in SPLITS:
        for class_name in CLASSES:
            directory = root / split / class_name
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(f"Unreadable crop image: {path}") from error
                records.append(
                    CropRecord(
                        path=path.resolve(),
                        source=source,
                        split=split,
                        class_name=class_name,
                        digest=_sha256(path),
                    )
                )
    return records


def _reject_leakage(records: list[CropRecord]) -> None:
    """
    Reject identical content assigned to different splits or classes.

    Parameters
    ----------
    records : list[CropRecord]
        Complete records from both input datasets.

    Raises
    ------
    ValueError
        Raised for cross-split or cross-class exact duplicates.
    """
    assignments: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        assignments[record.digest].add((record.split, record.class_name))
    conflicts = {
        digest: values for digest, values in assignments.items() if len(values) > 1
    }
    if conflicts:
        digest, values = next(iter(conflicts.items()))
        raise ValueError(
            f"Exact crop leakage for SHA-256 {digest}: {sorted(values)}"
        )


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Safely create or replace a generated output directory.

    Parameters
    ----------
    output : Path
        Target dataset root.
    replace : bool
        Whether marked generated data may be replaced.
    """
    marker = output / GENERATED_MARKER
    if output.exists():
        if not replace:
            raise ValueError(f"Output already exists: {output}")
        if not marker.is_file():
            raise ValueError(f"Refusing to replace unmarked directory: {output}")
        shutil.rmtree(output)
    for split in SPLITS:
        for class_name in CLASSES:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)
    marker.write_text("generated\n", encoding="utf-8")


def _quota(config: BalanceConfig, split: str) -> int:
    """
    Return the per-source class quota for a split.

    Parameters
    ----------
    config : BalanceConfig
        Dataset preparation configuration.
    split : str
        Classification split name.

    Returns
    -------
    int
        Requested crop count per source and class.
    """
    return {
        "train": config.train_per_source_class,
        "val": config.val_per_source_class,
        "test": config.test_per_source_class,
    }[split]


def _unique_by_digest(records: list[CropRecord]) -> list[CropRecord]:
    """
    Retain one deterministic record per exact image digest.

    Parameters
    ----------
    records : list[CropRecord]
        Candidate crop records.

    Returns
    -------
    list[CropRecord]
        Deduplicated candidates ordered by path.
    """
    unique: dict[str, CropRecord] = {}
    for record in sorted(records, key=lambda item: str(item.path)):
        unique.setdefault(record.digest, record)
    return list(unique.values())


def prepare_balanced_dataset(config: BalanceConfig) -> dict[str, Any]:
    """
    Build the source-balanced classifier dataset.

    Parameters
    ----------
    config : BalanceConfig
        Validated preparation configuration.

    Returns
    -------
    dict[str, Any]
        Reproducibility manifest with source and selection counts.
    """
    legacy = _records(config.legacy, "legacy")
    dataset3 = _records(config.dataset3, "dataset3")
    all_records = legacy + dataset3
    _reject_leakage(all_records)
    _prepare_output(config.output, config.replace)

    grouped: defaultdict[tuple[str, str, str], list[CropRecord]] = defaultdict(list)
    for record in all_records:
        grouped[(record.source, record.split, record.class_name)].append(record)

    random_generator = random.Random(config.seed)
    selected: list[CropRecord] = []
    selected_counts: Counter[tuple[str, str, str]] = Counter()
    overlap_counts: Counter[tuple[str, str]] = Counter()
    for split in SPLITS:
        for class_name in CLASSES:
            quota = _quota(config, split)
            legacy_candidates = _unique_by_digest(
                grouped[("legacy", split, class_name)]
            )
            random_generator.shuffle(legacy_candidates)
            if len(legacy_candidates) < quota:
                raise ValueError(
                    f"legacy/{split}/{class_name} provides {len(legacy_candidates)} "
                    f"unique crops, but {quota} are required"
                )
            legacy_selected = legacy_candidates[:quota]
            legacy_hashes = {record.digest for record in legacy_candidates}

            dataset3_candidates = _unique_by_digest(
                grouped[("dataset3", split, class_name)]
            )
            overlaps = [
                record for record in dataset3_candidates if record.digest in legacy_hashes
            ]
            overlap_counts[(split, class_name)] = len(overlaps)
            dataset3_candidates = [
                record
                for record in dataset3_candidates
                if record.digest not in legacy_hashes
            ]
            random_generator.shuffle(dataset3_candidates)
            if len(dataset3_candidates) < quota:
                raise ValueError(
                    f"dataset3/{split}/{class_name} provides "
                    f"{len(dataset3_candidates)} source-unique crops, but {quota} "
                    "are required"
                )
            dataset3_selected = dataset3_candidates[:quota]
            for record in legacy_selected + dataset3_selected:
                extension = record.path.suffix.casefold()
                filename = f"{record.source}_{record.digest[:20]}{extension}"
                destination = config.output / split / class_name / filename
                shutil.copy2(record.path, destination)
                selected.append(record)
                selected_counts[(record.source, split, class_name)] += 1

    selected_digests = [record.digest for record in selected]
    if len(selected_digests) != len(set(selected_digests)):
        raise RuntimeError("Generated source-balanced dataset contains exact duplicates")
    fingerprint = hashlib.sha256(
        "\n".join(sorted(selected_digests)).encode("utf-8")
    ).hexdigest().upper()
    manifest: dict[str, Any] = {
        "purpose": "source-balanced bolt/nut/other crop-classifier training",
        "sources": {
            "legacy": str(config.legacy.resolve()),
            "dataset3": str(config.dataset3.resolve()),
        },
        "configuration": {
            "train_per_source_class": config.train_per_source_class,
            "val_per_source_class": config.val_per_source_class,
            "test_per_source_class": config.test_per_source_class,
            "seed": config.seed,
        },
        "input_images": {
            "legacy": len(legacy),
            "dataset3": len(dataset3),
        },
        "cross_source_exact_overlaps_removed": {
            split: {
                class_name: overlap_counts[(split, class_name)]
                for class_name in CLASSES
            }
            for split in SPLITS
        },
        "selected_by_source_split_class": {
            source: {
                split: {
                    class_name: selected_counts[(source, split, class_name)]
                    for class_name in CLASSES
                }
                for split in SPLITS
            }
            for source in ("legacy", "dataset3")
        },
        "selected_totals": {
            split: {
                class_name: sum(
                    selected_counts[(source, split, class_name)]
                    for source in ("legacy", "dataset3")
                )
                for class_name in CLASSES
            }
            for split in SPLITS
        },
        "exact_duplicate_groups": 0,
        "cross_split_duplicate_groups": 0,
        "cross_class_duplicate_groups": 0,
        "dataset_fingerprint_sha256": fingerprint,
    }
    (config.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_config() -> BalanceConfig:
    """
    Parse and validate command-line configuration.

    Returns
    -------
    BalanceConfig
        Validated immutable configuration.
    """
    arguments = build_parser().parse_args()
    quotas = (
        arguments.train_per_source_class,
        arguments.val_per_source_class,
        arguments.test_per_source_class,
    )
    if any(value <= 0 for value in quotas):
        raise ValueError("All per-source class quotas must be positive")
    return BalanceConfig(
        legacy=arguments.legacy,
        dataset3=arguments.dataset3,
        output=arguments.output,
        train_per_source_class=arguments.train_per_source_class,
        val_per_source_class=arguments.val_per_source_class,
        test_per_source_class=arguments.test_per_source_class,
        seed=arguments.seed,
        replace=arguments.replace,
    )


def main() -> None:
    """Prepare balanced crops without starting model training."""
    manifest = prepare_balanced_dataset(parse_config())
    print(json.dumps(manifest, indent=2))
    print("Source-balanced crop dataset is ready. Training was not started.")


if __name__ == "__main__":
    main()
