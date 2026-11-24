"""Checks for Part 1: grayscale blocks, padding, and vector round-trip."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from part1_preprocessing import (
    parse_block_size,
    preprocess_array,
    preprocess_image,
    reconstruct_original,
)


class ParseBlockSizeTests(unittest.TestCase):
    def test_square_and_pair_forms(self) -> None:
        self.assertEqual(parse_block_size(4), (4, 4))
        self.assertEqual(parse_block_size("4"), (4, 4))
        self.assertEqual(parse_block_size("2x2"), (2, 2))
        self.assertEqual(parse_block_size("2×2"), (2, 2))
        self.assertEqual(parse_block_size("2*4"), (2, 4))
        self.assertEqual(parse_block_size((2, 4)), (2, 4))

    def test_rejects_invalid_sizes(self) -> None:
        with self.assertRaises(ValueError):
            parse_block_size(0)
        with self.assertRaises(ValueError):
            parse_block_size("2x")
        with self.assertRaises(ValueError):
            parse_block_size(-2)


class VectorizeTests(unittest.TestCase):
    def test_single_block_is_row_major(self) -> None:
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = preprocess_array(image, "2x2")

        self.assertEqual(result.original_shape, (2, 2))
        self.assertEqual(result.padded_shape, (2, 2))
        self.assertEqual(result.metadata.pad_bottom, 0)
        self.assertEqual(result.metadata.pad_right, 0)
        np.testing.assert_array_equal(result.vectors, np.array([[1, 2, 3, 4]], dtype=np.uint8))

    def test_non_overlapping_blocks_left_to_right_then_down(self) -> None:
        image = np.arange(8, dtype=np.uint8).reshape(2, 4)
        result = preprocess_array(image, (2, 2))

        self.assertEqual(result.vectors.shape, (2, 4))
        np.testing.assert_array_equal(result.vectors[0], [0, 1, 4, 5])
        np.testing.assert_array_equal(result.vectors[1], [2, 3, 6, 7])
        restored = reconstruct_original(result.vectors, result.metadata)
        np.testing.assert_array_equal(restored, image)

    def test_padding_when_size_is_not_divisible(self) -> None:
        image = np.arange(15, dtype=np.uint8).reshape(3, 5)
        result = preprocess_array(image, 2)

        self.assertEqual(result.original_shape, (3, 5))
        self.assertEqual(result.padded_shape, (4, 6))
        self.assertEqual(result.metadata.pad_bottom, 1)
        self.assertEqual(result.metadata.pad_right, 1)
        self.assertEqual(result.metadata.num_blocks, 6)
        self.assertEqual(result.metadata.vector_length, 4)

        restored = reconstruct_original(result.vectors, result.metadata)
        np.testing.assert_array_equal(restored, image)

    def test_block_larger_than_image_pads_to_one_block(self) -> None:
        image = np.array([[9, 8], [7, 6]], dtype=np.uint8)
        result = preprocess_array(image, "4x4")

        self.assertEqual(result.padded_shape, (4, 4))
        self.assertEqual(result.vectors.shape, (1, 16))
        self.assertEqual(int(result.vectors[0, 0]), 9)
        self.assertEqual(int(result.vectors[0, -1]), 0)
        restored = reconstruct_original(result.vectors, result.metadata)
        np.testing.assert_array_equal(restored, image)


class ImageFileTests(unittest.TestCase):
    def test_reads_color_image_as_grayscale_and_keeps_size(self) -> None:
        color = np.zeros((5, 7, 3), dtype=np.uint8)
        color[..., 0] = 255
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.png"
            Image.fromarray(color, mode="RGB").save(path)
            result = preprocess_image(path, "2x2")

        self.assertEqual(result.original_shape, (5, 7))
        self.assertEqual(result.padded_shape, (6, 8))
        self.assertEqual(result.vectors.dtype, np.uint8)
        self.assertEqual(result.vectors.shape, (12, 4))
        restored = reconstruct_original(result.vectors, result.metadata)
        self.assertEqual(restored.shape, (5, 7))
        self.assertGreater(int(restored.mean()), 0)


if __name__ == "__main__":
    unittest.main()
