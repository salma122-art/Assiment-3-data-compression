"""Part 1 — image processing and preprocessing for vector quantization."""

from .pipeline import (
    PreprocessResult,
    parse_block_size,
    preprocess_array,
    preprocess_image,
    reconstruct_original,
)
from .vectorizer import ImageMetadata

__all__ = [
    "ImageMetadata",
    "PreprocessResult",
    "parse_block_size",
    "preprocess_array",
    "preprocess_image",
    "reconstruct_original",
]
