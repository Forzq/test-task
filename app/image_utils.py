"""Image geometry helpers shared by training and API inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class SquareCropExtractor:
    """
    Extract square candidate crops while preserving object geometry.

    Parameters
    ----------
    output_size : int
        Width and height of every returned crop.
    padding_colour : tuple[int, int, int]
        RGB colour used when a square extends outside the source image.

    Notes
    -----
    The extractor is a reusable object because API inference and dataset
    preparation must apply exactly the same crop geometry.
    """

    output_size: int = 224
    padding_colour: tuple[int, int, int] = (114, 114, 114)

    def __post_init__(self) -> None:
        """
        Validate immutable crop settings immediately after construction.

        Raises
        ------
        ValueError
            Raised when the requested output size is not positive.
        """
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

    def extract(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        context_ratio: float = 0.20,
    ) -> Image.Image:
        """
        Extract one context-aware crop centred on a bounding box.

        Parameters
        ----------
        image : PIL.Image.Image
            Source RGB image.
        bbox : tuple[float, float, float, float]
            Candidate box in ``x1, y1, x2, y2`` pixel coordinates.
        context_ratio : float, optional
            Extra context added to every side relative to the longest box edge.

        Returns
        -------
        PIL.Image.Image
            Padded RGB crop resized to the configured square dimensions.

        Raises
        ------
        ValueError
            Raised for invalid geometry or a box outside the source image.
        """
        if context_ratio < 0.0:
            raise ValueError("context_ratio cannot be negative")

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError("bbox must have positive width and height")

        side = max(
            2,
            math.ceil(max(box_width, box_height) * (1.0 + 2.0 * context_ratio)),
        )
        centre_x = (x1 + x2) / 2.0
        centre_y = (y1 + y2) / 2.0
        square_left = math.floor(centre_x - side / 2.0)
        square_top = math.floor(centre_y - side / 2.0)
        square_right = square_left + side
        square_bottom = square_top + side

        source_left = max(0, square_left)
        source_top = max(0, square_top)
        source_right = min(image.width, square_right)
        source_bottom = min(image.height, square_bottom)
        if source_right <= source_left or source_bottom <= source_top:
            raise ValueError("bbox does not overlap the source image")

        canvas = Image.new("RGB", (side, side), self.padding_colour)
        region = image.crop(
            (source_left, source_top, source_right, source_bottom)
        ).convert("RGB")
        canvas.paste(region, (source_left - square_left, source_top - square_top))
        return canvas.resize(
            (self.output_size, self.output_size), Image.Resampling.LANCZOS
        )


def extract_square_crop(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    context_ratio: float = 0.20,
    output_size: int = 224,
) -> Image.Image:
    """
    Extract a square, context-aware crop without distorting the object.

    Parameters
    ----------
    image : PIL.Image.Image
        Source RGB image.
    bbox : tuple[float, float, float, float]
        Candidate box in ``x1, y1, x2, y2`` pixel coordinates.
    context_ratio : float, optional
        Extra context added to every side relative to the longest box edge.
    output_size : int, optional
        Width and height of the returned square image.

    Returns
    -------
    PIL.Image.Image
        RGB crop padded with a neutral grey colour and resized to a square.

    Raises
    ------
    ValueError
        Raised for an invalid box, context ratio, or output size.

    Notes
    -----
    Padding preserves long bolt geometry and keeps edge objects centred. This
    is preferable to stretching or centre-cropping the candidate.
    """
    return SquareCropExtractor(output_size=output_size).extract(
        image=image,
        bbox=bbox,
        context_ratio=context_ratio,
    )
