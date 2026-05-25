#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the raw nightdataset directory matches the expected filename manifest."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory that stores the raw nightdataset images.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to nightdataset_manifest.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest}")

    expected = {
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    actual = {path.name for path in input_dir.iterdir() if path.is_file()}

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing or unexpected:
        if missing:
            print("Missing files:")
            for name in missing:
                print(f"  {name}")
        if unexpected:
            print("Unexpected files:")
            for name in unexpected:
                print(f"  {name}")
        raise SystemExit(1)

    print("Manifest check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
