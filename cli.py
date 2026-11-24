"""Ask the user for an image path and a block size, then print Part 1 outputs."""

from pathlib import Path

from .pipeline import preprocess_image

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "samples" / "sample.png"


def main() -> None:
    print("Press Enter to use the sample image:")
    print(f"  {SAMPLE_IMAGE}")
    typed_path = input("Image path: ").strip().strip('"')
    image_path = SAMPLE_IMAGE if typed_path == "" else typed_path
    block_size = input("Block size (e.g. 2x2 or 4x4): ").strip()

    try:
        result = preprocess_image(image_path, block_size)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        print("Use a real file on this PC, or press Enter to use:")
        print(f"  {SAMPLE_IMAGE}")
        return

    meta = result.metadata

    print(f"Original dimensions: {meta.original_height} x {meta.original_width}")
    print(f"Padded dimensions:   {meta.padded_height} x {meta.padded_width}")
    print(f"Block size:          {meta.block_height} x {meta.block_width}")
    print(f"Padding (bottom, right): ({meta.pad_bottom}, {meta.pad_right})")
    print(f"Vectors:             {meta.num_blocks} x {meta.vector_length}")


if __name__ == "__main__":
    main()
