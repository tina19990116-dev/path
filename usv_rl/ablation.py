"""
Ablation runner to sweep auxiliary loss, attention, and recurrent components.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List

from .main import build_parser, train_loop
from .utils import ensure_dir


def variant_overrides(name: str) -> dict:
    if name == "no_aux":
        return {"no_aux": True}
    if name == "no_attn":
        return {"no_attn": True}
    if name == "ff_only":
        return {"ff_only": True}
    return {}


def run_ablation_suite(args) -> None:
    variants: List[str] = getattr(args, "variants", ["baseline"])
    results = []
    for name in variants:
        overrides = variant_overrides(name)
        variant_args = copy.deepcopy(args)
        for key, val in overrides.items():
            setattr(variant_args, key, val)
        variant_args.save_dir = str(ensure_dir(Path(args.save_dir) / name))
        print(f"\n=== Running ablation: {name} ===")
        train_loop(variant_args)
        results.append({"variant": name, "save_dir": variant_args.save_dir})
    summary_path = ensure_dir(args.save_dir) / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Ablation summary saved to {summary_path}")


def main():
    parser = build_parser()
    parser.description = "USV RL Ablations"
    parser.set_defaults(save_dir="outputs/ablations")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "no_aux", "no_attn", "ff_only"],
        help="Ablation variants to run sequentially.",
    )
    args = parser.parse_args()
    run_ablation_suite(args)


if __name__ == "__main__":
    main()

