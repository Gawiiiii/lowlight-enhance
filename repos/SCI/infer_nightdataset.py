from pathlib import Path

import argparse
import numpy as np
import torch
import torch.utils
from PIL import Image

from model_test import Network
from multi_read_data import MemoryFriendlyLoader


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]

DEFAULT_DATA_PATH = PROJECT_ROOT / "datasets" / "nightdataset"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "outputs" / "sci" / "nightdataset"
DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "demo.pt"


parser = argparse.ArgumentParser("sci-nightdataset")
parser.add_argument(
    "--data_path",
    type=str,
    default=str(DEFAULT_DATA_PATH),
    help="location of the campus night dataset",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=str(DEFAULT_SAVE_PATH),
    help="save location of the inference results",
)
parser.add_argument(
    "--model",
    type=str,
    default=str(DEFAULT_MODEL_PATH),
    help="path to the SCI pretrained weights",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="inference device, e.g. cuda or cpu",
)


def save_images(tensor, path):
    image_numpy = tensor[0].detach().cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    image.save(path, "PNG")


def load_model(model_path, device):
    model = Network().to(device)
    state_dict = torch.load(model_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"failed to load weights cleanly: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model


def main():
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    save_path = Path(args.save_path).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print("cuda is unavailable, falling back to cpu", flush=True)

    save_path.mkdir(parents=True, exist_ok=True)

    test_dataset = MemoryFriendlyLoader(img_dir=str(data_path), task="test")
    test_queue = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    model = load_model(str(model_path), device)

    with torch.inference_mode():
        for _, (input_tensor, image_name) in enumerate(test_queue):
            input_tensor = input_tensor.to(device, non_blocking=(device.type == "cuda"))
            stem = Path(image_name[0]).stem
            output = model(input_tensor)

            output_name = f"{stem}_enhance.png"
            output_path = save_path / output_name
            print(f"processing {output_name}", flush=True)
            save_images(output, output_path)


if __name__ == "__main__":
    main()
