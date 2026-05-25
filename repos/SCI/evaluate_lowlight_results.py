#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CSV_FIELDS = [
    "image_id",
    "input_filename",
    "output_filename",
    "width",
    "height",
    "input_brightness_mean",
    "output_brightness_mean",
    "brightness_gain_mean",
    "input_dark_ratio",
    "output_dark_ratio",
    "dark_ratio_change",
    "input_overexposure_ratio",
    "output_overexposure_ratio",
    "overexposure_ratio_change",
    "input_entropy",
    "output_entropy",
    "entropy_gain",
    "input_laplacian_var",
    "output_laplacian_var",
    "laplacian_gain",
    "mean_abs_delta_ab",
    "dark_mask_ratio",
    "dark_hf_noise_input",
    "dark_hf_noise_output",
    "dark_hf_noise_gain",
    "dark_noise_index_input",
    "dark_noise_index_output",
    "dark_noise_index_gain",
]

DARK_THRESHOLD = 0.15
HIGHLIGHT_THRESHOLD = 0.90
DARK_NOISE_INDEX_FLOOR = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified no-reference evaluation for low-light enhancement outputs."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of input images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory of enhanced images.")
    parser.add_argument(
        "--output-pattern",
        type=str,
        default="{stem}_enhance.png",
        help="Output filename pattern built from the input stem.",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="evaluation",
        help="Prefix for csv/json reports written into output-dir.",
    )
    return parser.parse_args()


def rgb_to_luma(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def as_cv_float(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.astype(np.float32))


def build_dark_mask(input_luma: np.ndarray) -> np.ndarray:
    quantile_threshold = float(np.quantile(input_luma, 0.25))
    threshold = min(0.20, quantile_threshold)
    mask = input_luma <= threshold
    if not np.any(mask):
        mask = input_luma <= float(np.quantile(input_luma, 0.10))
    return mask


def high_frequency_energy(luma: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(luma, (0, 0), sigmaX=1.2)
    return np.abs(luma - blurred)


def laplacian_variance(gray_uint8: np.ndarray) -> float:
    gray = gray_uint8.astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


def image_entropy(gray_uint8: np.ndarray) -> float:
    hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def collect_row(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_image = load_rgb(input_path)
    output_image = load_rgb(output_path)

    if input_image.shape != output_image.shape:
        raise ValueError(
            f"shape mismatch for {input_path.name}: input {input_image.shape} vs output {output_image.shape}"
        )

    input_luma = rgb_to_luma(input_image)
    output_luma = rgb_to_luma(output_image)
    input_gray_uint8 = np.clip(np.round(input_luma * 255.0), 0, 255).astype(np.uint8)
    output_gray_uint8 = np.clip(np.round(output_luma * 255.0), 0, 255).astype(np.uint8)

    input_lab = cv2.cvtColor(as_cv_float(input_image), cv2.COLOR_RGB2LAB)
    output_lab = cv2.cvtColor(as_cv_float(output_image), cv2.COLOR_RGB2LAB)
    delta_ab = output_lab[..., 1:] - input_lab[..., 1:]

    dark_mask = build_dark_mask(input_luma)
    dark_hf_input = high_frequency_energy(input_luma)
    dark_hf_output = high_frequency_energy(output_luma)

    if np.any(dark_mask):
        dark_hf_noise_input = float(dark_hf_input[dark_mask].mean())
        dark_hf_noise_output = float(dark_hf_output[dark_mask].mean())
        dark_luma_input = float(input_luma[dark_mask].mean())
        dark_luma_output = float(output_luma[dark_mask].mean())
        dark_mask_ratio = float(dark_mask.mean())
    else:
        dark_hf_noise_input = 0.0
        dark_hf_noise_output = 0.0
        dark_luma_input = 0.0
        dark_luma_output = 0.0
        dark_mask_ratio = 0.0

    dark_noise_index_input = dark_hf_noise_input / max(dark_luma_input, DARK_NOISE_INDEX_FLOOR)
    dark_noise_index_output = dark_hf_noise_output / max(dark_luma_output, DARK_NOISE_INDEX_FLOOR)
    input_entropy = image_entropy(input_gray_uint8)
    output_entropy = image_entropy(output_gray_uint8)
    input_laplacian = laplacian_variance(input_gray_uint8)
    output_laplacian = laplacian_variance(output_gray_uint8)

    return {
        "image_id": input_path.stem,
        "input_filename": input_path.name,
        "output_filename": output_path.name,
        "width": int(input_image.shape[1]),
        "height": int(input_image.shape[0]),
        "input_brightness_mean": float(input_luma.mean()),
        "output_brightness_mean": float(output_luma.mean()),
        "brightness_gain_mean": float(output_luma.mean() - input_luma.mean()),
        "input_dark_ratio": float((input_luma <= DARK_THRESHOLD).mean()),
        "output_dark_ratio": float((output_luma <= DARK_THRESHOLD).mean()),
        "dark_ratio_change": float((output_luma <= DARK_THRESHOLD).mean() - (input_luma <= DARK_THRESHOLD).mean()),
        "input_overexposure_ratio": float((input_luma >= HIGHLIGHT_THRESHOLD).mean()),
        "output_overexposure_ratio": float((output_luma >= HIGHLIGHT_THRESHOLD).mean()),
        "overexposure_ratio_change": float(
            (output_luma >= HIGHLIGHT_THRESHOLD).mean() - (input_luma >= HIGHLIGHT_THRESHOLD).mean()
        ),
        "input_entropy": input_entropy,
        "output_entropy": output_entropy,
        "entropy_gain": float(output_entropy - input_entropy),
        "input_laplacian_var": input_laplacian,
        "output_laplacian_var": output_laplacian,
        "laplacian_gain": float(output_laplacian - input_laplacian),
        "mean_abs_delta_ab": float(np.mean(np.abs(delta_ab))),
        "dark_mask_ratio": dark_mask_ratio,
        "dark_hf_noise_input": dark_hf_noise_input,
        "dark_hf_noise_output": dark_hf_noise_output,
        "dark_hf_noise_gain": float(dark_hf_noise_output - dark_hf_noise_input),
        "dark_noise_index_input": float(dark_noise_index_input),
        "dark_noise_index_output": float(dark_noise_index_output),
        "dark_noise_index_gain": float(dark_noise_index_output - dark_noise_index_input),
    }


def discover_inputs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file())


def build_output_lookup(output_dir: Path) -> dict[str, Path]:
    lookup = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        name = path.stem
        for suffix in ("_enhance", "_png_enhanced", "_jpg_enhanced", "_jpeg_enhanced", "_heic_enhanced", "_dng_enhanced"):
            if name.endswith(suffix):
                lookup[name[: -len(suffix)]] = path
                break
        else:
            lookup[name] = path
    return lookup


def write_reports(rows: list[dict[str, Any]], output_dir: Path, prefix: str, input_dir: Path) -> None:
    csv_path = output_dir / f"{prefix}_report.csv"
    json_path = output_dir / f"{prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = build_summary(rows, input_dir, output_dir)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def build_summary(rows: list[dict[str, Any]], input_dir: Path, output_dir: Path) -> dict[str, Any]:
    numeric_fields = [
        "input_brightness_mean",
        "output_brightness_mean",
        "brightness_gain_mean",
        "input_dark_ratio",
        "output_dark_ratio",
        "dark_ratio_change",
        "input_overexposure_ratio",
        "output_overexposure_ratio",
        "overexposure_ratio_change",
        "input_entropy",
        "output_entropy",
        "entropy_gain",
        "input_laplacian_var",
        "output_laplacian_var",
        "laplacian_gain",
        "mean_abs_delta_ab",
        "dark_hf_noise_gain",
        "dark_noise_index_gain",
    ]
    averages = {
        field: float(np.mean([float(row[field]) for row in rows])) if rows else None for field in numeric_fields
    }

    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images_evaluated": len(rows),
        "average_metrics": averages,
        "metric_notes": {
            "brightness_mean": "Mean luminance using Rec.709 luma.",
            "dark_ratio": f"Fraction of pixels with luma <= {DARK_THRESHOLD}. Lower after enhancement is usually better.",
            "overexposure_ratio": f"Fraction of pixels with luma >= {HIGHLIGHT_THRESHOLD}. Lower increase is safer.",
            "entropy": "Shannon entropy on 8-bit luma. Higher may indicate richer detail, but can also rise with noise.",
            "laplacian_var": "Variance of Laplacian on grayscale. Higher suggests stronger edges/texture, but may also reflect amplified noise.",
            "mean_abs_delta_ab": "Average absolute chroma shift in Lab space. Lower is usually more natural.",
            "dark_noise_index_gain": "Change in dark-region high-frequency energy normalized by dark-region luma. Lower is safer.",
        },
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir does not exist: {input_dir}")
    if not output_dir.exists():
        raise FileNotFoundError(f"output dir does not exist: {output_dir}")

    output_lookup = build_output_lookup(output_dir)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for input_path in discover_inputs(input_dir):
        output_path = output_lookup.get(input_path.stem)
        if output_path is None:
            missing.append(input_path.name)
            continue
        rows.append(collect_row(input_path, output_path))

    if missing:
        print(f"warning: {len(missing)} input images have no matched output")
        for name in missing[:10]:
            print(f"  missing: {name}")

    write_reports(rows, output_dir, args.report_prefix, input_dir)
    print(
        f"evaluated {len(rows)} image(s); report saved to "
        f"{output_dir / (args.report_prefix + '_report.csv')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
