#!/usr/bin/env python3
"""
Inspect and optionally export NPZ datasets for the marine environment simulator.

The script prints dataset structure, preview statistics, and can export arrays
to CSV files for easier manual inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Tuple

import numpy as np


def format_stats(array: np.ndarray) -> str:
    """Return a short statistics string for numeric arrays."""
    if array.size == 0:
        return "empty"
    flat = array.reshape(-1)
    return (
        f"min={flat.min():.4f}, max={flat.max():.4f}, "
        f"mean={flat.mean():.4f}, std={flat.std():.4f}"
    )


def preview_values(array: np.ndarray, limit: int) -> str:
    """Return a compact preview of the first few values."""
    if array.ndim == 0:
        return str(array)
    flat = array.reshape(-1)
    snippet = flat[:limit]
    ellipsis = " ..." if flat.size > limit else ""
    return f"{np.array2string(snippet, threshold=limit)}{ellipsis}"


def db_save_array(array: np.ndarray, path: str) -> None:
    """Save array to disk in CSV format (flattened for >2 dims)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_text = array.dtype.kind in {"U", "S", "O"}
    fmt = "%s" if is_text else "%.18e"
    if array.ndim <= 2:
        np.savetxt(path, array, delimiter=",", fmt=fmt)
    else:
        flat = array.reshape(array.shape[0], -1)
        np.savetxt(path, flat, delimiter=",", fmt=fmt)


def export_vector_field(
    field: np.ndarray,
    time_labels: Iterable[str],
    output_dir: str,
    prefix: str,
) -> None:
    """Export each (ny, nx, 2) vector slice into separate CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for idx, label in enumerate(time_labels):
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
        u = field[idx, :, :, 0]
        v = field[idx, :, :, 1]
        np.savetxt(os.path.join(output_dir, f"{prefix}_{safe_label}_u.csv"), u, delimiter=",")
        np.savetxt(os.path.join(output_dir, f"{prefix}_{safe_label}_v.csv"), v, delimiter=",")


def summarize_dataset(path: str, preview_limit: int) -> Tuple[np.lib.npyio.NpzFile, list[str]]:
    """Load and print dataset summary; return the opened NPZ and time labels."""
    data = np.load(path, allow_pickle=False)
    print(f"Loaded dataset: {path}")
    print("Keys:", data.files)
    print()
    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        if np.issubdtype(arr.dtype, np.number):
            print(" ", format_stats(arr))
        else:
            unique = np.unique(arr)
            sample = ", ".join(map(str, unique[:preview_limit]))
            if unique.size > preview_limit:
                sample += ", ..."
            print(f"  values: {sample}")
        print(f"  preview: {preview_values(arr, preview_limit)}")
        print()
    time_labels = [str(t) for t in data["time_labels"]] if "time_labels" in data.files else []
    return data, time_labels


def export_dataset(
    data: np.lib.npyio.NpzFile,
    time_labels: Iterable[str],
    output_dir: str,
    export_all: bool,
    export_fields: Iterable[str],
) -> None:
    """Export arrays to CSV files for manual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting arrays to '{output_dir}'...")
    keys = export_fields or data.files
    for key in keys:
        if key not in data.files:
            print(f"  Skipping missing key '{key}'.")
            continue
        arr = data[key]
        if arr.ndim == 4 and arr.shape[-1] == 2 and key in {"ocean_currents", "wind_field"}:
            export_vector_field(arr, time_labels, output_dir, key)
            print(f"  Exported vector field '{key}' slices.")
        elif export_all or arr.ndim <= 2:
            out_path = os.path.join(output_dir, f"{key}.csv")
            db_save_array(arr, out_path)
            print(f"  Saved '{key}' to {out_path}.")
        else:
            print(
                f"  Skipped '{key}' (shape={arr.shape}); use --export-all to flatten high-dimensional data."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and export NPZ marine environment datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        default="outputs/mock_environment_dataset.npz",
        help="Path to the NPZ dataset.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=5,
        help="Number of elements to show in previews.",
    )
    parser.add_argument(
        "--export-dir",
        help="Directory to export arrays as CSV files.",
    )
    parser.add_argument(
        "--export-fields",
        nargs="*",
        default=[],
        help="Specific dataset keys to export. Defaults to all keys when --export-dir is provided.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="For non-selected keys, export flattened arrays even if they exceed two dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Dataset '{args.file}' was not found.")

    data, time_labels = summarize_dataset(args.file, args.preview_limit)

    if args.export_dir:
        export_dataset(
            data,
            time_labels,
            args.export_dir,
            export_all=args.export_all,
            export_fields=args.export_fields,
        )
    data.close()


if __name__ == "__main__":
    main()
