"""Pad an image so both sides divide evenly by the block size, and crop that pad later."""

import numpy as np


def padding_amounts(height: int, width: int, block_height: int, block_width: int) -> tuple[int, int]:
    """Return (pad_bottom, pad_right) needed for non-overlapping blocks."""
    pad_bottom = (block_height - (height % block_height)) % block_height
    pad_right = (block_width - (width % block_width)) % block_width
    return pad_bottom, pad_right


def pad_to_block_size(
    image: np.ndarray,
    block_height: int,
    block_width: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Pad on the bottom and right. Returns (padded_image, pad_bottom, pad_right)."""
    height, width = image.shape
    pad_bottom, pad_right = padding_amounts(height, width, block_height, block_width)
    if pad_bottom == 0 and pad_right == 0:
        return image, 0, 0

    padded = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )
    return padded, pad_bottom, pad_right


def remove_padding(image: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    """Drop padding added on the bottom and right, restoring the original size."""
    if original_height <= 0 or original_width <= 0:
        raise ValueError("Original dimensions must be positive.")
    if original_height > image.shape[0] or original_width > image.shape[1]:
        raise ValueError("Original dimensions are larger than the padded image.")
    return image[:original_height, :original_width]
