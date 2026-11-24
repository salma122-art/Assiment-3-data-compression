"""Part 1 pipeline: image -> grayscale -> pad -> non-overlapping vectors + metadata."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .image_reader import load_image
from .padding import pad_to_block_size, remove_padding
from .vectorizer import ImageMetadata, blocks_to_vectors, vectors_to_image


@dataclass(frozen=True)
class PreprocessResult:
    """What Member 2 receives from Part 1."""

    vectors: np.ndarray
    metadata: ImageMetadata

    @property
    def original_shape(self) -> tuple[int, int]:
        return self.metadata.original_height, self.metadata.original_width

    @property
    def padded_shape(self) -> tuple[int, int]:
        return self.metadata.padded_height, self.metadata.padded_width


def parse_block_size(block_size: int | str | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Accept 4, '4', '2x2', '2×2', '2*2', or (2, 2). A single number means a square block."""
    if isinstance(block_size, tuple) or isinstance(block_size, list):
        if len(block_size) != 2:
            raise ValueError("Block size tuple must be (height, width).")
        height, width = int(block_size[0]), int(block_size[1])
    elif isinstance(block_size, int) and not isinstance(block_size, bool):
        height = width = block_size
    elif isinstance(block_size, str):
        text = block_size.strip().lower().replace("×", "x").replace("*", "x")
        if "x" in text:
            parts = text.split("x")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid block size: {block_size!r}. Use 2x2 or 4.")
            height, width = int(parts[0]), int(parts[1])
        else:
            height = width = int(text)
    else:
        raise ValueError(f"Invalid block size: {block_size!r}. Use 2x2 or 4.")

    if height <= 0 or width <= 0:
        raise ValueError("Block size must be a positive integer.")
    return height, width


def preprocess_array(
    image: np.ndarray,
    block_size: int | str | tuple[int, int] | list[int],
    pad_value: int = 0,
) -> PreprocessResult:
    """Turn a grayscale image array into vectors and the metadata needed to undo padding."""
    if image.ndim != 2:
        raise ValueError("Expected a grayscale image with shape (height, width).")
    if image.size == 0:
        raise ValueError("Image is empty.")

    pixels = np.asarray(image)
    block_height, block_width = parse_block_size(block_size)
    original_height, original_width = pixels.shape
    padded, pad_bottom, pad_right = pad_to_block_size(
        pixels, block_height, block_width, pad_value=pad_value
    )
    vectors = blocks_to_vectors(padded, block_height, block_width)
    padded_height, padded_width = padded.shape

    metadata = ImageMetadata(
        original_height=int(original_height),
        original_width=int(original_width),
        padded_height=int(padded_height),
        padded_width=int(padded_width),
        block_height=block_height,
        block_width=block_width,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        pad_value=int(pad_value),
        num_blocks_vertical=padded_height // block_height,
        num_blocks_horizontal=padded_width // block_width,
    )
    return PreprocessResult(vectors=vectors, metadata=metadata)


def preprocess_image(
    image_path: str | Path,
    block_size: int | str | tuple[int, int] | list[int],
    pad_value: int = 0,
) -> PreprocessResult:
    """Read an image, convert it to grayscale, and extract block vectors."""
    image = load_image(image_path)
    return preprocess_array(image, block_size, pad_value=pad_value)


def reconstruct_original(vectors: np.ndarray, metadata: ImageMetadata) -> np.ndarray:
    """Rebuild the image from vectors and crop the padding recorded in metadata."""
    padded = vectors_to_image(vectors, metadata)
    return remove_padding(padded, metadata.original_height, metadata.original_width)
