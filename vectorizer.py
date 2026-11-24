"""Split a padded image into non-overlapping blocks and flatten each block to a vector."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageMetadata:
    """Sizes Member 2 and decompression need, including how to drop padding."""

    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    block_height: int
    block_width: int
    pad_bottom: int
    pad_right: int
    pad_value: int
    num_blocks_vertical: int
    num_blocks_horizontal: int

    @property
    def num_blocks(self) -> int:
        return self.num_blocks_vertical * self.num_blocks_horizontal

    @property
    def vector_length(self) -> int:
        return self.block_height * self.block_width

    def as_dict(self) -> dict:
        return {
            "original_height": self.original_height,
            "original_width": self.original_width,
            "padded_height": self.padded_height,
            "padded_width": self.padded_width,
            "block_height": self.block_height,
            "block_width": self.block_width,
            "pad_bottom": self.pad_bottom,
            "pad_right": self.pad_right,
            "pad_value": self.pad_value,
            "num_blocks_vertical": self.num_blocks_vertical,
            "num_blocks_horizontal": self.num_blocks_horizontal,
            "num_blocks": self.num_blocks,
            "vector_length": self.vector_length,
        }


def blocks_to_vectors(padded_image: np.ndarray, block_height: int, block_width: int) -> np.ndarray:
    """Return vectors of shape (num_blocks, block_height * block_width), row-major blocks."""
    height, width = padded_image.shape
    if height % block_height != 0 or width % block_width != 0:
        raise ValueError("Image dimensions must be divisible by the block size before vectorizing.")

    rows = height // block_height
    cols = width // block_width
    blocks = padded_image.reshape(rows, block_height, cols, block_width)
    ordered = blocks.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(ordered.reshape(rows * cols, block_height * block_width))


def vectors_to_image(vectors: np.ndarray, metadata: ImageMetadata) -> np.ndarray:
    """Rebuild the padded image from vectors. Block order matches blocks_to_vectors."""
    expected = (metadata.num_blocks, metadata.vector_length)
    if vectors.shape != expected:
        raise ValueError(f"Expected vectors of shape {expected}, got {vectors.shape}.")

    rows = metadata.num_blocks_vertical
    cols = metadata.num_blocks_horizontal
    block_height = metadata.block_height
    block_width = metadata.block_width
    blocks = vectors.reshape(rows, cols, block_height, block_width)
    ordered = blocks.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(
        ordered.reshape(metadata.padded_height, metadata.padded_width)
    )
