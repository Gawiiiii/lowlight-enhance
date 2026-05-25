#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

try:
    import rawpy
except Exception:
    rawpy = None


SUPPORTED_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".dng"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize a folder of low-light images so that the long edge equals the given target."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with raw input images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory used to store PNG outputs.")
    parser.add_argument("--long-edge", type=int, default=2560, help="Target long edge.")
    return parser.parse_args()


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".dng":
        if rawpy is None:
            raise RuntimeError("rawpy is required to read DNG images.")
        with rawpy.imread(str(path)) as raw:
            image = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        return Image.fromarray(image)

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image.copy()


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    width, height = image.size
    current_long_edge = max(width, height)
    if current_long_edge == long_edge:
        return image

    scale = long_edge / float(current_long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = load_rgb(image_path)
        resized = resize_long_edge(image, args.long_edge)
        relative_path = image_path.relative_to(input_dir)
        save_path = output_dir / relative_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(save_path)
        print(f"saved {save_path}")

    print(f"Finished resizing {len(image_paths)} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
