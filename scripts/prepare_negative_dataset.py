"""Download and prepare diverse empty-label images for YOLO background training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
CALTECH101_URL = (
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
)
CALTECH101_MD5 = "3138e1922a9193bfa496528edbbc45d0"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")
EXCLUDED_CALTECH_CATEGORIES = {"background_google"}


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    """
    Statistics for one prepared negative dataset.

    Parameters
    ----------
    split_counts : dict[str, int]
        Number of empty-label images assigned to each YOLO split.
    source_counts : dict[str, int]
        Number of unique images retained from each internet or local source.
    """

    split_counts: dict[str, int]
    source_counts: dict[str, int]


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for reproducible negative-data preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with official COCO128 and Caltech 101 sources enabled by default.
    """
    parser = argparse.ArgumentParser(
        description="Create a diverse YOLO dataset whose label files are intentionally empty."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/negatives"),
        help="Output directory containing train, valid, and test negative splits.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Deliberately replace a previously prepared negative output directory.",
    )
    parser.add_argument(
        "--generic-source",
        type=Path,
        action="append",
        default=[],
        help="Additional pre-vetted image directory; can be specified more than once.",
    )
    parser.add_argument(
        "--download-coco128",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the 128-image COCO sample (default: enabled).",
    )
    parser.add_argument(
        "--coco-directory",
        type=Path,
        default=Path("data/raw_negatives/coco128"),
        help="Cache directory for the downloaded COCO128 archive.",
    )
    parser.add_argument(
        "--download-caltech101",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a diverse sample from Caltech 101 (default: enabled).",
    )
    parser.add_argument(
        "--caltech-directory",
        type=Path,
        default=Path("data/raw_negatives/caltech-101"),
        help="Cache directory for the downloaded Caltech 101 archive.",
    )
    parser.add_argument(
        "--caltech-count",
        type=int,
        default=500,
        help="Number of unique Caltech 101 images to include (default: 500).",
    )
    return parser


def _image_files(directory: Path) -> list[Path]:
    """
    Return supported image files from a directory tree in stable order.

    Parameters
    ----------
    directory : Path
        Root directory to scan recursively.

    Returns
    -------
    list[Path]
        Sorted image paths with recognized file suffixes.

    Raises
    ------
    ValueError
        Raised when the requested source directory does not exist.
    """
    if not directory.is_dir():
        raise ValueError(f"Image directory was not found: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _file_digest(path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate a file digest without loading the full image into memory.

    Parameters
    ----------
    path : Path
        File whose bytes will be hashed.
    algorithm : str
        Hash algorithm accepted by ``hashlib``.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """
    Extract a ZIP archive while rejecting paths outside the destination.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open archive to extract.
    destination : Path
        Directory into which members may be written.

    Raises
    ------
    ValueError
        Raised when an archive member attempts path traversal.
    """
    resolved_destination = destination.resolve()
    for member in archive.infolist():
        target = (resolved_destination / member.filename).resolve()
        if resolved_destination not in target.parents and target != resolved_destination:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(resolved_destination)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    """
    Extract a TAR archive while rejecting links and paths outside the destination.

    Parameters
    ----------
    archive : tarfile.TarFile
        Open TAR or TAR.GZ archive to extract.
    destination : Path
        Directory into which regular members may be written.

    Raises
    ------
    ValueError
        Raised when a member is a link or attempts path traversal.
    """
    resolved_destination = destination.resolve()
    for member in archive.getmembers():
        target = (resolved_destination / member.name).resolve()
        if resolved_destination not in target.parents and target != resolved_destination:
            raise ValueError(f"Unsafe TAR archive member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not allowed: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read archive member: {member.name}")
            with source, target.open("wb") as destination_file:
                shutil.copyfileobj(source, destination_file)
        else:
            raise ValueError(f"Unsupported TAR archive member: {member.name}")


def _download_zip(
    url: str,
    archive_path: Path,
    extraction_directory: Path,
    expected_md5: str | None = None,
) -> None:
    """
    Download, validate, and safely extract one public ZIP archive.

    Parameters
    ----------
    url : str
        Public source URL.
    archive_path : Path
        Temporary local ZIP path.
    extraction_directory : Path
        Destination directory for extracted files.
    expected_md5 : str, optional
        Publisher-provided MD5 digest used to verify the download.

    Raises
    ------
    ValueError
        Raised when publisher checksum validation fails.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive_path)
    if expected_md5 and _file_digest(archive_path, "md5") != expected_md5:
        raise ValueError(f"Checksum validation failed for {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, extraction_directory)
    archive_path.unlink(missing_ok=True)


def _coco_images(destination: Path) -> list[Path]:
    """
    Download COCO128 when necessary and return its 128 source images.

    Parameters
    ----------
    destination : Path
        Expected extracted ``coco128`` directory.

    Returns
    -------
    list[Path]
        COCO128 image paths.

    Raises
    ------
    ValueError
        Raised when extraction does not create the expected image directory.
    """
    image_directory = destination / "images" / "train2017"
    if not image_directory.is_dir():
        _download_zip(
            COCO128_URL,
            destination.parent / "coco128.zip",
            destination.parent,
        )
    if not image_directory.is_dir():
        raise ValueError(f"COCO128 images were not extracted to {image_directory}")
    return _image_files(image_directory)


def _caltech_category_root(destination: Path) -> Path:
    """
    Download Caltech 101 when necessary and locate its category directories.

    Parameters
    ----------
    destination : Path
        Expected extracted ``caltech-101`` directory.

    Returns
    -------
    Path
        Directory whose children are object-category folders.

    Raises
    ------
    ValueError
        Raised when extraction does not contain ``101_ObjectCategories``.
    """
    category_root = destination / "101_ObjectCategories"
    if not category_root.is_dir():
        nested_archive = destination / "101_ObjectCategories.tar.gz"
        if not nested_archive.is_file():
            _download_zip(
                CALTECH101_URL,
                destination.parent / "caltech-101.zip",
                destination.parent,
                expected_md5=CALTECH101_MD5,
            )
        if nested_archive.is_file() and not category_root.is_dir():
            with tarfile.open(nested_archive, "r:gz") as archive:
                _safe_extract_tar(archive, destination)
    if not category_root.is_dir():
        candidates = list(destination.parent.rglob("101_ObjectCategories"))
        if len(candidates) == 1:
            category_root = candidates[0]
    if not category_root.is_dir():
        raise ValueError("Caltech 101 archive has no 101_ObjectCategories directory")
    return category_root


def _balanced_caltech_images(category_root: Path, count: int) -> list[Path]:
    """
    Select unique images round-robin across Caltech object categories.

    Parameters
    ----------
    category_root : Path
        Directory containing one subdirectory per Caltech category.
    count : int
        Exact number of distinct images to select.

    Returns
    -------
    list[Path]
        Content-unique images spread across as many categories as possible.

    Raises
    ------
    ValueError
        Raised when the source cannot provide the requested number of unique files.
    """
    if count <= 0:
        raise ValueError("caltech-count must be positive")
    categories = [
        directory
        for directory in sorted(category_root.iterdir())
        if directory.is_dir() and directory.name.lower() not in EXCLUDED_CALTECH_CATEGORIES
    ]
    queues = {directory: iter(_image_files(directory)) for directory in categories}
    selected: list[Path] = []
    seen_digests: set[str] = set()

    while queues and len(selected) < count:
        exhausted: list[Path] = []
        for category, images in queues.items():
            try:
                image = next(images)
            except StopIteration:
                exhausted.append(category)
                continue
            digest = _file_digest(image)
            if digest not in seen_digests:
                selected.append(image)
                seen_digests.add(digest)
                if len(selected) == count:
                    break
        for category in exhausted:
            queues.pop(category)

    if len(selected) != count:
        raise ValueError(
            f"Caltech 101 provided {len(selected)} unique images; requested {count}."
        )
    return selected


def _deduplicate_sources(sources: list[tuple[str, Iterable[Path]]]) -> list[tuple[str, Path, str]]:
    """
    Remove byte-identical images across all configured input sources.

    Parameters
    ----------
    sources : list[tuple[str, Iterable[Path]]]
        Named source collections in preferred retention order.

    Returns
    -------
    list[tuple[str, Path, str]]
        Source name, unique path, and SHA-256 content digest for each retained image.
    """
    retained: list[tuple[str, Path, str]] = []
    seen: set[str] = set()
    for source_name, image_paths in sources:
        for image_path in image_paths:
            digest = _file_digest(image_path)
            if digest in seen:
                continue
            retained.append((source_name, image_path, digest))
            seen.add(digest)
    return retained


def _split_for_digest(digest: str) -> str:
    """
    Assign an image digest to a deterministic 80/10/10 split.

    Parameters
    ----------
    digest : str
        SHA-256 image-content digest.

    Returns
    -------
    str
        One of ``train``, ``valid``, or ``test``.
    """
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def _copy_as_negative(
    image_path: Path,
    digest: str,
    output_root: Path,
) -> str:
    """
    Copy one image and create its deliberately empty YOLO label file.

    Parameters
    ----------
    image_path : Path
        Vetted source image containing no target annotation.
    digest : str
        Content digest used for split assignment and collision-free naming.
    output_root : Path
        Prepared negative dataset root.

    Returns
    -------
    str
        Split name to which the image was assigned.
    """
    split = _split_for_digest(digest)
    name = f"negative_{digest[:16]}{image_path.suffix.lower()}"
    image_directory = output_root / split / "images"
    label_directory = output_root / split / "labels"
    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_directory / name)
    (label_directory / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
    return split


def _prepare_output(output_root: Path, replace: bool) -> None:
    """
    Validate or deliberately clear the generated negative-data directory.

    Parameters
    ----------
    output_root : Path
        Intended generated-data destination.
    replace : bool
        Whether an existing generated directory may be removed.

    Raises
    ------
    ValueError
        Raised when generated data already exists without ``--replace``.
    """
    if output_root.exists() and any(output_root.rglob("*")):
        if not replace:
            raise ValueError(
                f"Negative output already exists: {output_root}. "
                "Pass --replace to rebuild this generated directory deliberately."
            )
        shutil.rmtree(output_root)


def _write_manifest(
    output_root: Path,
    summary: PreparationSummary,
) -> None:
    """
    Record source provenance and final counts beside generated data.

    Parameters
    ----------
    output_root : Path
        Prepared negative dataset root.
    summary : PreparationSummary
        Counts to persist for review and reproducibility.
    """
    payload = {
        "sources": {
            "coco128": COCO128_URL,
            "caltech101": CALTECH101_URL,
        },
        "source_counts": summary.source_counts,
        "split_counts": summary.split_counts,
        "annotation_policy": "Every label file is intentionally empty: no nut or bolt.",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def prepare_dataset(arguments: argparse.Namespace) -> PreparationSummary:
    """
    Download, deduplicate, split, and label configured generic images.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command-line source and output options.

    Returns
    -------
    PreparationSummary
        Counts for each internet source and generated YOLO split.

    Raises
    ------
    ValueError
        Raised when no image source is enabled or output cannot be rebuilt safely.
    """
    sources: list[tuple[str, Iterable[Path]]] = []

    if arguments.download_coco128:
        sources.append(("coco128", _coco_images(arguments.coco_directory)))
    if arguments.download_caltech101:
        category_root = _caltech_category_root(arguments.caltech_directory)
        sources.append(
            (
                "caltech101",
                _balanced_caltech_images(category_root, arguments.caltech_count),
            )
        )
    for index, source in enumerate(arguments.generic_source, start=1):
        sources.append((f"custom_{index}", _image_files(source)))
    if not sources:
        raise ValueError("Enable an internet source or provide --generic-source")

    unique_images = _deduplicate_sources(sources)
    _prepare_output(arguments.output, arguments.replace)
    split_counts = {split: 0 for split in SPLITS}
    source_counts: dict[str, int] = {}
    for source_name, image_path, digest in unique_images:
        split_counts[_copy_as_negative(image_path, digest, arguments.output)] += 1
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

    summary = PreparationSummary(split_counts=split_counts, source_counts=source_counts)
    _write_manifest(arguments.output, summary)
    return summary


def main() -> None:
    """
    Build the negative dataset and print verifiable source and split counts.

    Raises
    ------
    ValueError
        Raised when downloads, uniqueness checks, or output validation fail.
    """
    summary = prepare_dataset(build_parser().parse_args())
    print(f"Negative sources: {summary.source_counts}")
    print(f"Negative splits: {summary.split_counts}")


if __name__ == "__main__":
    main()
