"""Mine unrelated images that the current detector mistakes for nuts or bolts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TARGET_CLASSES = {"bolt", "nut"}


@dataclass(frozen=True, slots=True)
class HardNegative:
    """
    False-positive candidate selected by the current detector.

    Parameters
    ----------
    source : str
        Original image path.
    confidence : float
        Highest nut or bolt confidence on the image.
    predicted_class : str
        Class name of the highest-confidence false detection.
    """

    source: str
    confidence: float
    predicted_class: str


def build_parser() -> argparse.ArgumentParser:
    """
    Create the hard-negative mining command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with project-local defaults.
    """
    parser = argparse.ArgumentParser(
        description="Find high-confidence nut/bolt mistakes in a known-negative image pool."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_negatives/caltech-101/101_ObjectCategories"),
        help="Recursive directory known not to contain nuts or bolts.",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("models/best.pt"), help="Current YOLO checkpoint."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hard_negatives"),
        help="Generated empty-label YOLO dataset.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=Path("data/negatives"),
        help="Existing negative dataset whose content hashes must not be duplicated.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum mined images.")
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=3000,
        help="Maximum number of reproducibly sampled candidates to inspect.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of paths passed to Ultralytics at once to bound RAM and VRAM use.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Candidate sampling seed.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.20,
        help="Minimum false-detection confidence considered for ranking.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--device", default="0", help="Ultralytics inference device.")
    parser.add_argument("--replace", action="store_true", help="Replace generated output.")
    return parser


def _image_paths(root: Path) -> list[Path]:
    """
    Recursively enumerate supported image files.

    Parameters
    ----------
    root : Path
        Candidate pool root.

    Returns
    -------
    list[Path]
        Sorted candidate image paths.
    """
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


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


def _excluded_digest_prefixes(root: Path) -> set[str]:
    """
    Read digest prefixes embedded by the generic-negative preparation script.

    Parameters
    ----------
    root : Path
        Existing prepared negative dataset.

    Returns
    -------
    set[str]
        Sixteen-character SHA-256 prefixes already used by training.
    """
    prefixes: set[str] = set()
    if not root.is_dir():
        return prefixes
    for image_path in _image_paths(root):
        if image_path.stem.startswith("negative_"):
            prefixes.add(image_path.stem.removeprefix("negative_")[:16])
        else:
            prefixes.add(_sha256(image_path)[:16])
    return prefixes


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create an empty generated YOLO dataset.

    Parameters
    ----------
    output : Path
        Destination root.
    replace : bool
        Whether an existing generated directory may be removed.
    """
    if output.exists():
        if not replace:
            raise FileExistsError(f"Output already exists; use --replace: {output}")
        shutil.rmtree(output)
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)


def _mine(arguments: argparse.Namespace) -> list[HardNegative]:
    """
    Run the current model on candidates and return ranked false positives.

    Parameters
    ----------
    arguments : argparse.Namespace
        Validated command-line options.

    Returns
    -------
    list[HardNegative]
        Highest-confidence unique false-positive images.
    """
    from ultralytics import YOLO

    excluded = _excluded_digest_prefixes(arguments.exclude)
    candidates = [
        path for path in _image_paths(arguments.source) if _sha256(path)[:16] not in excluded
    ]
    random.Random(arguments.seed).shuffle(candidates)
    candidates = candidates[: arguments.scan_limit]
    model = YOLO(str(arguments.weights))
    mined: list[HardNegative] = []
    for start in range(0, len(candidates), arguments.chunk_size):
        chunk = candidates[start : start + arguments.chunk_size]
        results = model.predict(
            source=[str(path) for path in chunk],
            conf=arguments.min_confidence,
            imgsz=arguments.imgsz,
            batch=arguments.batch,
            device=arguments.device,
            stream=False,
            verbose=False,
        )
        for image_path, result in zip(chunk, results, strict=True):
            if result.boxes is None:
                continue
            detections: list[tuple[float, str]] = []
            for detection in result.boxes:
                class_name = str(result.names[int(detection.cls[0].item())]).lower()
                if class_name in TARGET_CLASSES:
                    detections.append((float(detection.conf[0].item()), class_name))
            if detections:
                confidence, class_name = max(detections)
                mined.append(HardNegative(str(image_path.resolve()), confidence, class_name))
    return sorted(mined, key=lambda item: item.confidence, reverse=True)[: arguments.limit]


def _write_dataset(
    output: Path, mined: list[HardNegative], arguments: argparse.Namespace
) -> None:
    """
    Copy selected images with deliberately empty label files.

    Parameters
    ----------
    output : Path
        Generated dataset root.
    mined : list[HardNegative]
        Ranked false-positive images.
    arguments : argparse.Namespace
        Mining settings recorded for reproducibility.
    """
    for item in mined:
        source = Path(item.source)
        digest = _sha256(source)
        name = f"hard_negative_{digest[:16]}{source.suffix.lower()}"
        shutil.copy2(source, output / "train" / "images" / name)
        (output / "train" / "labels" / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
    data_yaml = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": ["bolt", "nut"],
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    report = {
        "weights": str(arguments.weights.resolve()),
        "source": str(arguments.source.resolve()),
        "minimum_confidence": arguments.min_confidence,
        "scanned_candidates": arguments.scan_limit,
        "selected": len(mined),
        "items": [asdict(item) for item in mined],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    """
    Mine and persist a reproducible hard-negative training source.
    """
    arguments = build_parser().parse_args()
    if not arguments.source.is_dir() or not arguments.weights.is_file():
        raise ValueError("Source directory and model weights must exist")
    if min(arguments.limit, arguments.scan_limit, arguments.chunk_size, arguments.batch) <= 0:
        raise ValueError("limits, chunk-size, and batch must be positive")
    if not 0.0 < arguments.min_confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    _prepare_output(arguments.output, arguments.replace)
    mined = _mine(arguments)
    _write_dataset(arguments.output, mined, arguments)
    print(f"Mined hard negatives: {len(mined)}")
    if mined:
        print(
            f"Confidence range: {mined[-1].confidence:.4f}..{mined[0].confidence:.4f}"
        )
    print(f"Generated dataset: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
