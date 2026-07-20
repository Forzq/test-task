"""Convert mixed YOLO polygon annotations to axis-aligned detection boxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_classifier_dataset import (
    _dataset_entries,
    _image_paths,
    _label_path,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Create the annotation-normalisation parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser supporting safe dry-run and explicit apply modes.
    """
    parser = argparse.ArgumentParser(
        description="Convert polygon label lines to enclosing YOLO xywh boxes."
    )
    parser.add_argument("--data", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write converted labels; without this flag only report changes.",
    )
    return parser


def _convert_line(line: str) -> tuple[str, bool]:
    """
    Convert one polygon label line to a normalised YOLO box.

    Parameters
    ----------
    line : str
        One stripped YOLO annotation line.

    Returns
    -------
    tuple[str, bool]
        Converted line and whether a polygon conversion occurred.
    """
    values = line.split()
    if len(values) <= 5:
        return line, False
    coordinates = [float(value) for value in values[1:]]
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError(f"Malformed polygon line: {line}")
    x_values = coordinates[0::2]
    y_values = coordinates[1::2]
    x1, x2 = min(x_values), max(x_values)
    y1, y2 = min(y_values), max(y_values)
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    converted = (
        f"{values[0]} {centre_x:.8f} {centre_y:.8f} "
        f"{width:.8f} {height:.8f}"
    )
    return converted, True


def normalize(data_path: Path, apply_changes: bool) -> tuple[int, int]:
    """
    Find and optionally convert every polygon annotation in a dataset YAML.

    Parameters
    ----------
    data_path : Path
        Combined YOLO dataset YAML.
    apply_changes : bool
        Whether converted label content should be written.

    Returns
    -------
    tuple[int, int]
        Number of modified files and converted polygon lines.
    """
    label_paths: set[Path] = set()
    for roots in _dataset_entries(data_path).values():
        for image_root in roots:
            label_paths.update(
                _label_path(image_path, image_root)
                for image_path in _image_paths(image_root)
            )

    changed_files = 0
    converted_lines = 0
    for label_path in sorted(label_paths):
        original_lines = label_path.read_text(encoding="utf-8").splitlines()
        output_lines: list[str] = []
        file_changed = False
        for original_line in original_lines:
            if not original_line.strip():
                continue
            converted, changed = _convert_line(original_line.strip())
            output_lines.append(converted)
            file_changed |= changed
            converted_lines += int(changed)
        if file_changed:
            changed_files += 1
            print(label_path)
            if apply_changes:
                label_path.write_text(
                    "\n".join(output_lines) + "\n", encoding="utf-8"
                )
    return changed_files, converted_lines


def main() -> None:
    """Report or apply polygon-to-box annotation conversion."""
    arguments = build_parser().parse_args()
    if not arguments.data.is_file():
        raise ValueError(f"Dataset YAML was not found: {arguments.data}")
    files, lines = normalize(arguments.data, arguments.apply)
    mode = "Converted" if arguments.apply else "Would convert"
    print(f"{mode} {lines} polygon lines in {files} label files.")


if __name__ == "__main__":
    main()
