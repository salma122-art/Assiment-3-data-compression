"""Read an input image and convert it to grayscale."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk and return a grayscale uint8 array of shape (H, W)."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        grayscale = image.convert("L")
        pixels = np.asarray(grayscale, dtype=np.uint8)

    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError(f"Could not read a usable image from: {path}")
    return pixels
