import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".dng",
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parents[1]

    parser = argparse.ArgumentParser(
        description="Run RetinexFormer on an arbitrary low-light image folder."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=project_root / "datasets" / "nightdataset",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "outputs" / "retinexformer" / "nightdataset",
        help="Directory for enhanced outputs.",
    )
    parser.add_argument(
        "--opt",
        type=Path,
        default=None,
        help="Path to an option YAML file. Defaults to LOL_v2_real, then LOL_v1.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to a checkpoint file. Defaults to LOL_v2_real, then LOL_v1.",
    )
    parser.add_argument(
        "--pad_factor",
        type=int,
        default=4,
        help="Pad inputs to a multiple of this factor before inference.",
    )
    parser.add_argument(
        "--split_threshold",
        type=int,
        default=3000,
        help="Run tiled inference when padded height or width exceeds this threshold.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1536,
        help="Tile size for large-image inference.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=64,
        help="Overlap between neighboring tiles during large-image inference.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index. Ignored when CUDA is unavailable.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=repo_root,
        help="Repository root. Usually auto-detected.",
    )
    return parser.parse_args()


def choose_default_paths(repo_root: Path):
    candidates = [
        (
            repo_root / "Options" / "RetinexFormer_LOL_v2_real.yml",
            repo_root / "pretrain_model" / "LOL_v2_real.pth",
        ),
        (
            repo_root / "Options" / "RetinexFormer_LOL_v2_real.yml",
            repo_root / "pretrained_weights" / "LOL_v2_real.pth",
        ),
        (
            repo_root / "Options" / "RetinexFormer_LOL_v1.yml",
            repo_root / "pretrain_model" / "LOL_v1.pth",
        ),
        (
            repo_root / "Options" / "RetinexFormer_LOL_v1.yml",
            repo_root / "pretrained_weights" / "LOL_v1.pth",
        ),
    ]

    for opt_path, weight_path in candidates:
        if opt_path.is_file() and weight_path.is_file():
            return opt_path, weight_path

    missing = "\n".join(f"- {opt_path}\n- {weight_path}" for opt_path, weight_path in candidates)
    raise FileNotFoundError(
        "Could not find a preferred low-light checkpoint/config pair.\n"
        "Expected one of:\n"
        f"{missing}"
    )


def resolve_paths(args):
    if args.opt is not None and args.weights is not None:
        return args.opt, args.weights

    default_opt, default_weights = choose_default_paths(args.repo_root)
    opt_path = args.opt or default_opt
    weights_path = args.weights or default_weights
    return opt_path, weights_path


def collect_images(input_dir: Path):
    paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def load_rgb_image(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".dng":
        import rawpy

        with rawpy.imread(str(path)) as raw:
            image = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        return image.astype(np.float32) / 255.0

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0


def save_rgb_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0.0, 1.0)
    image = np.rint(image * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def load_model(opt_path: Path, weights_path: Path, device):
    import torch
    from basicsr.models import create_model
    from basicsr.utils.options import parse

    opt = parse(str(opt_path), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.type == "cuda" else 0
    model = create_model(opt).net_g

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    params = checkpoint.get("params", checkpoint.get("state_dict", checkpoint))

    load_errors = []
    for state_dict in (
        params,
        {f"module.{k}": v for k, v in params.items()},
        {k.removeprefix("module."): v for k, v in params.items()},
    ):
        try:
            model.load_state_dict(state_dict, strict=True)
            break
        except RuntimeError as exc:
            load_errors.append(str(exc))
    else:
        raise RuntimeError(
            "Failed to load checkpoint.\n" + "\n\n".join(load_errors)
        )

    model = model.to(device)
    model.eval()
    return model


def pad_to_factor(image_tensor, factor: int):
    import torch.nn.functional as F

    _, _, h, w = image_tensor.shape
    padded_h = ((h + factor - 1) // factor) * factor
    padded_w = ((w + factor - 1) // factor) * factor
    pad_h = padded_h - h
    pad_w = padded_w - w
    if pad_h == 0 and pad_w == 0:
        return image_tensor, h, w
    padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, h, w


def run_model(model, input_tensor):
    return model(input_tensor)


def build_blend_weights(height: int, width: int, device):
    import torch

    y = torch.linspace(-1.0, 1.0, steps=height, device=device)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device)
    wy = 0.5 * (torch.cos(torch.pi * y) + 1.0)
    wx = 0.5 * (torch.cos(torch.pi * x) + 1.0)
    weight = wy[:, None] * wx[None, :]
    weight = weight.clamp_min(1e-3)
    return weight.unsqueeze(0).unsqueeze(0)


def forward_model(model, input_tensor, split_threshold: int, tile_size: int, tile_overlap: int):
    import torch

    _, _, h, w = input_tensor.shape
    if h <= split_threshold and w <= split_threshold:
        return run_model(model, input_tensor)

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size")

    output = torch.zeros_like(input_tensor)
    weight = torch.zeros_like(input_tensor)

    h_starts = list(range(0, max(h - tile_size, 0) + 1, stride))
    w_starts = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not h_starts or h_starts[-1] != max(h - tile_size, 0):
        h_starts.append(max(h - tile_size, 0))
    if not w_starts or w_starts[-1] != max(w - tile_size, 0):
        w_starts.append(max(w - tile_size, 0))

    for top in h_starts:
        for left in w_starts:
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            tile = input_tensor[:, :, top:bottom, left:right]
            restored_tile = run_model(model, tile)
            blend = build_blend_weights(bottom - top, right - left, restored_tile.device)
            output[:, :, top:bottom, left:right] += restored_tile * blend
            weight[:, :, top:bottom, left:right] += blend

    return output / weight.clamp_min(1e-6)


def main():
    args = parse_args()

    import torch
    from pillow_heif import register_heif_opener
    from tqdm import tqdm

    register_heif_opener()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    opt_path, weights_path = resolve_paths(args)
    opt_path = opt_path.expanduser().resolve()
    weights_path = weights_path.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not opt_path.is_file():
        raise FileNotFoundError(f"Config not found: {opt_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {input_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Using config: {opt_path}")
    print(f"Using weights: {weights_path}")
    print(f"Reading images from: {input_dir}")
    print(f"Saving results to: {output_dir}")
    print(f"Found {len(image_paths)} images")

    model = load_model(opt_path, weights_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc="Enhancing", unit="image"):
            image = load_rgb_image(image_path)
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            tensor, original_h, original_w = pad_to_factor(tensor, args.pad_factor)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            restored = forward_model(
                model,
                tensor,
                args.split_threshold,
                args.tile_size,
                args.tile_overlap,
            )
            restored = restored[:, :, :original_h, :original_w]
            restored = restored.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

            relative_path = image_path.relative_to(input_dir)
            save_path = output_dir / relative_path.with_suffix(".png")
            save_rgb_image(save_path, restored)

    print("Inference finished.")


if __name__ == "__main__":
    main()
