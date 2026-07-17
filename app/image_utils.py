"""Image geometry helpers shared by training and API inference."""

from __future__ import annotations

import math

from PIL import Image


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
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if context_ratio < 0.0:
        raise ValueError("context_ratio cannot be negative")

    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("bbox must have positive width and height")

    side = max(2, math.ceil(max(box_width, box_height) * (1.0 + 2.0 * context_ratio)))
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

    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    region = image.crop((source_left, source_top, source_right, source_bottom)).convert("RGB")
    canvas.paste(region, (source_left - square_left, source_top - square_top))
    return canvas.resize((output_size, output_size), Image.Resampling.LANCZOS)
