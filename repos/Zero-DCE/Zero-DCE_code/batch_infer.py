#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

import model

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
	import pillow_heif

	pillow_heif.register_heif_opener()
	HEIF_AVAILABLE = True
	HEIF_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local runtime
	HEIF_AVAILABLE = False
	HEIF_IMPORT_ERROR = str(exc)

try:
	import rawpy

	RAWPY_AVAILABLE = True
	RAWPY_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local runtime
	rawpy = None
	RAWPY_AVAILABLE = False
	RAWPY_IMPORT_ERROR = str(exc)


SUPPORTED_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".dng", ".png"}
DEFAULT_MAX_FULL_IMAGE_MEGAPIXELS = 16.0
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 32
MIN_TILE_SIZE = 256

BRIGHTNESS_IMPROVEMENT_THRESHOLD = 0.03
COLOR_SHIFT_THRESHOLD = 8.0
HIGHLIGHT_GAIN_THRESHOLD = 0.005
HIGHLIGHT_SCALE_THRESHOLD = 1.5
DARK_NOISE_GAIN_THRESHOLD = 0.002
DARK_NOISE_INDEX_FLOOR = 0.05
DARK_NOISE_INDEX_GAIN_THRESHOLD = 0.10

CSV_FIELDS = [
	"input_filename",
	"output_filename",
	"success",
	"inference_time_ms",
	"failure_reason",
	"image_width",
	"image_height",
	"input_luma_mean",
	"input_luma_median",
	"input_luma_std",
	"input_luma_p05",
	"input_luma_p95",
	"input_highlight_ratio",
	"input_shadow_ratio",
	"output_luma_mean",
	"output_luma_median",
	"output_luma_std",
	"output_luma_p05",
	"output_luma_p95",
	"output_highlight_ratio",
	"output_shadow_ratio",
	"brightness_gain_mean",
	"brightness_gain_median",
	"brightness_gain_p95",
	"mean_delta_e",
	"mean_abs_delta_ab",
	"highlight_ratio_gain",
	"highlight_area_scale",
	"dark_mask_ratio",
	"dark_luma_input",
	"dark_luma_output",
	"dark_hf_noise_input",
	"dark_hf_noise_output",
	"dark_hf_noise_gain",
	"dark_noise_index_input",
	"dark_noise_index_output",
	"dark_noise_index_gain",
	"brightness_improved",
	"possible_color_shift",
	"possible_highlight_expansion",
	"possible_dark_noise_amplification",
]


def parse_args() -> argparse.Namespace:
	code_dir = Path(__file__).resolve().parent
	repo_root = code_dir.parent
	project_root = repo_root.parents[1]

	parser = argparse.ArgumentParser(
		description="Batch inference for Zero-DCE with per-image logging and simple quality heuristics."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=project_root / "datasets" / "nightdataset",
		help="Directory that contains input images. The script scans recursively.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=project_root / "outputs" / "zero_dce" / "nightdataset",
		help="Directory used to save enhanced images and reports.",
	)
	parser.add_argument(
		"--weights",
		type=Path,
		default=code_dir / "snapshots" / "Epoch99.pth",
		help="Path to the Zero-DCE checkpoint.",
	)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Torch device, for example: cuda, cuda:0, cpu.",
	)
	parser.add_argument(
		"--max-full-image-megapixels",
		type=float,
		default=DEFAULT_MAX_FULL_IMAGE_MEGAPIXELS,
		help="Images larger than this use tiled inference automatically.",
	)
	parser.add_argument(
		"--tile-size",
		type=int,
		default=DEFAULT_TILE_SIZE,
		help="Core tile size used for tiled inference.",
	)
	parser.add_argument(
		"--tile-overlap",
		type=int,
		default=DEFAULT_TILE_OVERLAP,
		help="Tile overlap in pixels used to avoid seams.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	input_dir = args.input_dir.resolve()
	output_dir = args.output_dir.resolve()
	weights_path = args.weights.resolve()

	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
	if not weights_path.exists():
		raise FileNotFoundError(f"Checkpoint does not exist: {weights_path}")

	device = torch.device(args.device)
	net = load_model(weights_path, device)

	image_paths = discover_images(input_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	rows: list[dict[str, Any]] = []
	total_count = len(image_paths)
	print(f"Found {total_count} image(s) under {input_dir}")

	with torch.inference_mode():
		for index, image_path in enumerate(image_paths, start=1):
			row = process_image(
				net=net,
				image_path=image_path,
				input_dir=input_dir,
				output_dir=output_dir,
				device=device,
				max_full_image_megapixels=args.max_full_image_megapixels,
				tile_size=args.tile_size,
				tile_overlap=args.tile_overlap,
			)
			rows.append(row)
			status = "OK" if row["success"] else "FAIL"
			print(f"[{index}/{total_count}] {status} {row['input_filename']}")

	write_reports(
		rows=rows,
		output_dir=output_dir,
		input_dir=input_dir,
		weights_path=weights_path,
		device=device,
	)

	success_count = sum(1 for row in rows if row["success"])
	failure_count = total_count - success_count
	print(
		f"Finished. Success: {success_count}, Failure: {failure_count}. "
		f"Reports saved to {output_dir}"
	)
	return 0


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
	net = model.enhance_net_nopool().to(device)
	state_dict = torch.load(weights_path, map_location=device)
	net.load_state_dict(state_dict)
	net.eval()
	return net


def discover_images(input_dir: Path) -> list[Path]:
	return sorted(
		path
		for path in input_dir.rglob("*")
		if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
	)


def process_image(
	net: torch.nn.Module,
	image_path: Path,
	input_dir: Path,
	output_dir: Path,
	device: torch.device,
	max_full_image_megapixels: float,
	tile_size: int,
	tile_overlap: int,
) -> dict[str, Any]:
	base_row = {
		"input_filename": str(image_path.relative_to(input_dir)),
		"output_filename": "",
		"success": False,
		"inference_time_ms": None,
		"failure_reason": "",
		"image_width": None,
		"image_height": None,
	}
	base_row.update({field: None for field in CSV_FIELDS if field not in base_row})

	try:
		input_array = load_image_as_rgb(image_path)
		height, width = input_array.shape[:2]
		base_row["image_width"] = width
		base_row["image_height"] = height

		if device.type == "cuda":
			torch.cuda.synchronize(device)
		start = time.perf_counter()
		enhanced_tensor = infer_image(
			net=net,
			input_array=input_array,
			device=device,
			max_full_image_megapixels=max_full_image_megapixels,
			tile_size=tile_size,
			tile_overlap=tile_overlap,
		)
		if device.type == "cuda":
			torch.cuda.synchronize(device)
		base_row["inference_time_ms"] = round((time.perf_counter() - start) * 1000.0, 3)

		output_array = tensor_to_numpy(enhanced_tensor)
		output_path = build_output_path(image_path, input_dir, output_dir)
		save_rgb_image(output_array, output_path)

		metrics = compute_metrics(input_array, output_array)
		base_row.update(metrics)
		base_row["output_filename"] = str(output_path.relative_to(output_dir))
		base_row["success"] = True
		return base_row
	except Exception as exc:
		base_row["failure_reason"] = str(exc)
		return base_row


def load_image_as_rgb(image_path: Path) -> np.ndarray:
	suffix = image_path.suffix.lower()

	if suffix == ".dng":
		return load_dng_image(image_path)

	try:
		with Image.open(image_path) as image:
			rgb = ImageOps.exif_transpose(image).convert("RGB")
			return pil_to_numpy(rgb)
	except UnidentifiedImageError as exc:
		if suffix == ".heic" and not HEIF_AVAILABLE:
			raise RuntimeError(
				"HEIC decoding is unavailable. Install pillow-heif to read .heic files."
			) from exc
		raise


def load_dng_image(image_path: Path) -> np.ndarray:
	if RAWPY_AVAILABLE:
		with rawpy.imread(str(image_path)) as raw:
			rgb = raw.postprocess(
				use_camera_wb=True,
				no_auto_bright=True,
				output_bps=8,
			)
		return pil_to_numpy(Image.fromarray(rgb))

	cv_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
	if cv_image is not None:
		if cv_image.ndim == 2:
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
		elif cv_image.shape[2] == 4:
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
		else:
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
		return cv_image.astype(np.float32) / 255.0

	reason = "DNG decoding is unavailable."
	if RAWPY_IMPORT_ERROR:
		reason = f"{reason} Install rawpy to improve DNG support."
	raise RuntimeError(reason)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
	array = np.asarray(image, dtype=np.float32) / 255.0
	if array.ndim != 3 or array.shape[2] != 3:
		raise ValueError(f"Expected an RGB image, but got shape {array.shape}")
	return array


def numpy_to_tensor(image: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
	tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
	if device is not None:
		tensor = tensor.to(device=device)
	return tensor


def infer_image(
	net: torch.nn.Module,
	input_array: np.ndarray,
	device: torch.device,
	max_full_image_megapixels: float,
	tile_size: int,
	tile_overlap: int,
) -> torch.Tensor:
	input_tensor_cpu = numpy_to_tensor(input_array)
	height, width = input_array.shape[:2]
	image_megapixels = (height * width) / 1_000_000.0
	use_tiled_first = image_megapixels > max_full_image_megapixels

	if use_tiled_first:
		return infer_tiled(net, input_tensor_cpu, device=device, tile_size=tile_size, tile_overlap=tile_overlap)

	try:
		return infer_full_image(net, input_tensor_cpu, device=device)
	except RuntimeError as exc:
		if not is_oom_error(exc):
			raise
		if device.type == "cuda":
			torch.cuda.empty_cache()
		return infer_tiled(net, input_tensor_cpu, device=device, tile_size=tile_size, tile_overlap=tile_overlap)


def infer_full_image(
	net: torch.nn.Module,
	input_tensor_cpu: torch.Tensor,
	device: torch.device,
) -> torch.Tensor:
	input_tensor = input_tensor_cpu.to(device=device)
	try:
		_, enhanced_tensor, _ = net(input_tensor)
		return enhanced_tensor.detach().cpu()
	finally:
		del input_tensor


def infer_tiled(
	net: torch.nn.Module,
	input_tensor_cpu: torch.Tensor,
	device: torch.device,
	tile_size: int,
	tile_overlap: int,
) -> torch.Tensor:
	current_tile_size = tile_size
	last_error: RuntimeError | None = None

	while current_tile_size >= MIN_TILE_SIZE:
		try:
			return infer_tiled_once(
				net,
				input_tensor_cpu,
				device=device,
				tile_size=current_tile_size,
				tile_overlap=tile_overlap,
			)
		except RuntimeError as exc:
			if not is_oom_error(exc):
				raise
			last_error = exc
			if device.type == "cuda":
				torch.cuda.empty_cache()
			current_tile_size //= 2

	if last_error is not None:
		raise last_error
	raise RuntimeError("Tiled inference failed before any tile size could be attempted.")


def infer_tiled_once(
	net: torch.nn.Module,
	input_tensor_cpu: torch.Tensor,
	device: torch.device,
	tile_size: int,
	tile_overlap: int,
) -> torch.Tensor:
	_, _, height, width = input_tensor_cpu.shape
	output_tensor = torch.zeros((1, 3, height, width), dtype=torch.float32)

	for top in range(0, height, tile_size):
		bottom = min(top + tile_size, height)
		for left in range(0, width, tile_size):
			right = min(left + tile_size, width)

			padded_top = max(0, top - tile_overlap)
			padded_bottom = min(height, bottom + tile_overlap)
			padded_left = max(0, left - tile_overlap)
			padded_right = min(width, right + tile_overlap)

			input_patch = input_tensor_cpu[
				:,
				:,
				padded_top:padded_bottom,
				padded_left:padded_right,
			].to(device=device)
			try:
				_, enhanced_patch, _ = net(input_patch)
				enhanced_patch = enhanced_patch.detach().cpu()
			finally:
				del input_patch

			crop_top = top - padded_top
			crop_bottom = crop_top + (bottom - top)
			crop_left = left - padded_left
			crop_right = crop_left + (right - left)
			output_tensor[:, :, top:bottom, left:right] = enhanced_patch[
				:,
				:,
				crop_top:crop_bottom,
				crop_left:crop_right,
			]
			del enhanced_patch

			if device.type == "cuda":
				torch.cuda.empty_cache()

	return output_tensor


def is_oom_error(exc: RuntimeError) -> bool:
	message = str(exc).lower()
	return "out of memory" in message or "cuda error: out of memory" in message


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
	array = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
	return np.clip(array, 0.0, 1.0).astype(np.float32)


def build_output_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
	relative_parent = image_path.relative_to(input_dir).parent
	suffix_name = image_path.suffix.lower().lstrip(".")
	output_name = f"{image_path.stem}_{suffix_name}_enhanced.png"
	output_path = output_dir / relative_parent / output_name
	output_path.parent.mkdir(parents=True, exist_ok=True)
	return output_path


def save_rgb_image(image: np.ndarray, output_path: Path) -> None:
	image_uint8 = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
	Image.fromarray(image_uint8).save(output_path)


def compute_metrics(input_image: np.ndarray, output_image: np.ndarray) -> dict[str, Any]:
	input_luma = rgb_to_luma(input_image)
	output_luma = rgb_to_luma(output_image)

	input_stats = collect_luma_stats(input_luma, prefix="input")
	output_stats = collect_luma_stats(output_luma, prefix="output")

	input_lab = cv2.cvtColor(as_cv_float(input_image), cv2.COLOR_RGB2LAB)
	output_lab = cv2.cvtColor(as_cv_float(output_image), cv2.COLOR_RGB2LAB)

	delta_lab = output_lab - input_lab
	delta_ab = delta_lab[..., 1:]
	delta_e = np.linalg.norm(delta_lab, axis=2)

	dark_mask = build_dark_mask(input_luma)
	dark_hf_input = high_frequency_energy(input_luma)
	dark_hf_output = high_frequency_energy(output_luma)

	if np.any(dark_mask):
		dark_luma_input = float(input_luma[dark_mask].mean())
		dark_luma_output = float(output_luma[dark_mask].mean())
		dark_input_value = float(dark_hf_input[dark_mask].mean())
		dark_output_value = float(dark_hf_output[dark_mask].mean())
		dark_mask_ratio = float(dark_mask.mean())
	else:
		dark_luma_input = 0.0
		dark_luma_output = 0.0
		dark_input_value = 0.0
		dark_output_value = 0.0
		dark_mask_ratio = 0.0

	highlight_gain = output_stats["output_highlight_ratio"] - input_stats["input_highlight_ratio"]
	highlight_scale = ratio_or_none(
		output_stats["output_highlight_ratio"],
		input_stats["input_highlight_ratio"],
	)
	dark_noise_gain = dark_output_value - dark_input_value
	dark_noise_index_input = dark_input_value / max(dark_luma_input, DARK_NOISE_INDEX_FLOOR)
	dark_noise_index_output = dark_output_value / max(dark_luma_output, DARK_NOISE_INDEX_FLOOR)
	dark_noise_index_gain = dark_noise_index_output - dark_noise_index_input

	metrics = {
		**input_stats,
		**output_stats,
		"brightness_gain_mean": float(output_stats["output_luma_mean"] - input_stats["input_luma_mean"]),
		"brightness_gain_median": float(
			output_stats["output_luma_median"] - input_stats["input_luma_median"]
		),
		"brightness_gain_p95": float(output_stats["output_luma_p95"] - input_stats["input_luma_p95"]),
		"mean_delta_e": float(delta_e.mean()),
		"mean_abs_delta_ab": float(np.mean(np.abs(delta_ab))),
		"highlight_ratio_gain": float(highlight_gain),
		"highlight_area_scale": highlight_scale,
		"dark_mask_ratio": dark_mask_ratio,
		"dark_luma_input": dark_luma_input,
		"dark_luma_output": dark_luma_output,
		"dark_hf_noise_input": dark_input_value,
		"dark_hf_noise_output": dark_output_value,
		"dark_hf_noise_gain": float(dark_noise_gain),
		"dark_noise_index_input": float(dark_noise_index_input),
		"dark_noise_index_output": float(dark_noise_index_output),
		"dark_noise_index_gain": float(dark_noise_index_gain),
	}

	metrics["brightness_improved"] = metrics["brightness_gain_mean"] > BRIGHTNESS_IMPROVEMENT_THRESHOLD
	metrics["possible_color_shift"] = metrics["mean_abs_delta_ab"] > COLOR_SHIFT_THRESHOLD
	metrics["possible_highlight_expansion"] = (
		metrics["highlight_ratio_gain"] > HIGHLIGHT_GAIN_THRESHOLD
		and (
			metrics["highlight_area_scale"] is None
			or metrics["highlight_area_scale"] > HIGHLIGHT_SCALE_THRESHOLD
		)
	)
	metrics["possible_dark_noise_amplification"] = (
		metrics["dark_hf_noise_gain"] > DARK_NOISE_GAIN_THRESHOLD
		and metrics["dark_noise_index_gain"] > DARK_NOISE_INDEX_GAIN_THRESHOLD
	)
	return metrics


def collect_luma_stats(luma: np.ndarray, prefix: str) -> dict[str, float]:
	return {
		f"{prefix}_luma_mean": float(luma.mean()),
		f"{prefix}_luma_median": float(np.median(luma)),
		f"{prefix}_luma_std": float(luma.std()),
		f"{prefix}_luma_p05": float(np.quantile(luma, 0.05)),
		f"{prefix}_luma_p95": float(np.quantile(luma, 0.95)),
		f"{prefix}_highlight_ratio": float((luma >= 0.90).mean()),
		f"{prefix}_shadow_ratio": float((luma <= 0.15).mean()),
	}


def rgb_to_luma(image: np.ndarray) -> np.ndarray:
	return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


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


def ratio_or_none(numerator: float, denominator: float) -> float | None:
	if math.isclose(denominator, 0.0, abs_tol=1e-8):
		return None
	return float(numerator / denominator)


def as_cv_float(image: np.ndarray) -> np.ndarray:
	return np.ascontiguousarray(image.astype(np.float32))


def write_reports(
	rows: list[dict[str, Any]],
	output_dir: Path,
	input_dir: Path,
	weights_path: Path,
	device: torch.device,
) -> None:
	csv_path = output_dir / "inference_report.csv"
	with csv_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
		writer.writeheader()
		for row in rows:
			writer.writerow({field: row.get(field) for field in CSV_FIELDS})

	summary_path = output_dir / "inference_summary.json"
	summary = build_summary(rows, input_dir=input_dir, output_dir=output_dir, weights_path=weights_path, device=device)
	with summary_path.open("w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2, ensure_ascii=False)


def build_summary(
	rows: list[dict[str, Any]],
	input_dir: Path,
	output_dir: Path,
	weights_path: Path,
	device: torch.device,
) -> dict[str, Any]:
	success_rows = [row for row in rows if row["success"]]

	numeric_fields = [
		"input_luma_mean",
		"output_luma_mean",
		"brightness_gain_mean",
		"mean_abs_delta_ab",
		"highlight_ratio_gain",
		"dark_hf_noise_gain",
		"dark_noise_index_gain",
		"inference_time_ms",
	]
	averages = {}
	for field in numeric_fields:
		values = [float(row[field]) for row in success_rows if row.get(field) is not None]
		averages[field] = float(np.mean(values)) if values else None

	return {
		"created_at": datetime.now().astimezone().isoformat(),
		"input_dir": str(input_dir),
		"output_dir": str(output_dir),
		"weights": str(weights_path),
		"device": str(device),
		"supported_extensions": sorted(SUPPORTED_EXTENSIONS),
		"dependency_status": {
			"pillow_heif_available": HEIF_AVAILABLE,
			"rawpy_available": RAWPY_AVAILABLE,
			"pillow_heif_import_error": HEIF_IMPORT_ERROR,
			"rawpy_import_error": RAWPY_IMPORT_ERROR,
		},
		"totals": {
			"images_found": len(rows),
			"success": len(success_rows),
			"failure": len(rows) - len(success_rows),
		},
		"average_metrics_on_success": averages,
		"flag_counts": {
			"brightness_improved": sum(bool(row.get("brightness_improved")) for row in success_rows),
			"possible_color_shift": sum(bool(row.get("possible_color_shift")) for row in success_rows),
			"possible_highlight_expansion": sum(
				bool(row.get("possible_highlight_expansion")) for row in success_rows
			),
			"possible_dark_noise_amplification": sum(
				bool(row.get("possible_dark_noise_amplification")) for row in success_rows
			),
		},
		"heuristic_notes": {
			"brightness_improved": f"brightness_gain_mean > {BRIGHTNESS_IMPROVEMENT_THRESHOLD}",
			"possible_color_shift": f"mean_abs_delta_ab > {COLOR_SHIFT_THRESHOLD}",
			"possible_highlight_expansion": (
				f"highlight_ratio_gain > {HIGHLIGHT_GAIN_THRESHOLD} and "
				f"highlight_area_scale > {HIGHLIGHT_SCALE_THRESHOLD} when the input already has highlights"
			),
			"possible_dark_noise_amplification": (
				f"dark_hf_noise_gain > {DARK_NOISE_GAIN_THRESHOLD} and "
				f"dark_noise_index_gain > {DARK_NOISE_INDEX_GAIN_THRESHOLD}"
			),
			"dark_region_definition": "input luma <= min(0.20, 25th-percentile luma), fallback to the darkest 10%",
			"highlight_definition": "luma >= 0.90",
			"dark_noise_index_definition": (
				f"dark_hf_noise / max(dark_luma_mean, {DARK_NOISE_INDEX_FLOOR})"
			),
		},
	}


if __name__ == "__main__":
	raise SystemExit(main())
