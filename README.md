# Vector Quantization

Team submission folder. Rename it to `VectorQ_ID1_ID2_ID3` before upload.

| Part | Folder | Status |
| --- | --- | --- |
| 1 — image preprocessing | `part1_preprocessing/` | implemented |
| 2 — codebook | `part2_codebook/` | empty |
| 3 — compress / decompress | `part3_compression/` | empty |
| TA menu (compress or decompress) | `main.py` | empty |

## Part 1

Reads an image, converts it to grayscale, pads it so the size divides by the block, and splits it into non-overlapping 1D vectors.

```text
py -3 -m part1_preprocessing.cli
```

Press Enter at the image path to use `samples/sample.png`. A path like `C:\Users\hp\Pictures\test.png` only works if that file is actually on the PC.

Member 2 should call:

```python
from part1_preprocessing import preprocess_image

result = preprocess_image("image.png", "4x4")
result.vectors    # shape (num_blocks, block_height * block_width)
result.metadata   # original size, padded size, pad amounts
```

Padding is constant 0 on the bottom and right. `metadata.original_height` and `metadata.original_width` are what decompression uses to crop the pad. `reconstruct_original(vectors, metadata)` does that crop for a lossless round-trip of Part 1 (no quantization yet).
