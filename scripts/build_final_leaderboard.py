"""Concatenate per-phase leaderboards into outputs/eval/final_leaderboard.csv.

Sources:
  - outputs/eval/leaderboard_phase3.csv  (5 methods x 60 images x 3 ratios = 900 rows)
  - outputs/eval/lora_stage_b_gate.csv   (stage-B LoRA, 60 images, 5x only)
  - outputs/eval/leaderboard_phase4_6.csv (SUPIR / HYPIR / HYPIR-caption,
    12 subset images, 4x only; also has the 5 baseline methods at 4x for
    those 12 images, which we deduplicate against Phase 3)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "outputs" / "eval"


def main() -> int:
    p3 = pd.read_csv(EVAL_DIR / "leaderboard_phase3.csv")
    p3["source"] = "phase3"
    print(f"phase3:    {len(p3)} rows", file=sys.stderr)

    lora_path = EVAL_DIR / "lora_stage_b_gate.csv"
    lora_rows: pd.DataFrame
    if lora_path.is_file():
        lora_raw = pd.read_csv(lora_path)
        # Reshape: lora_stage_b_gate has lpips_lora + lpips_baseline_twostage + lpips_realesrgan side-by-side.
        # Extract just the LoRA row (others are duplicates of phase3 columns).
        lora_rows = lora_raw[
            ["image", "category", "subcategory", "challenges", "method", "lr_size", "lpips_lora"]
        ].rename(columns={"lpips_lora": "lpips"})
        lora_rows["dists"] = float("nan")
        lora_rows["ratio"] = 1000 // lora_rows["lr_size"]
        lora_rows["source"] = "phase4c_lora"
        print(f"phase4c lora: {len(lora_rows)} rows", file=sys.stderr)
    else:
        lora_rows = pd.DataFrame()

    p46_path = EVAL_DIR / "leaderboard_phase4_6.csv"
    p46_new: pd.DataFrame
    if p46_path.is_file():
        p46_raw = pd.read_csv(p46_path)
        # Only keep the new methods (supir/hypir/hypir_caption); the 5 baselines
        # for the 12 subset images at 4x are already in phase3.
        p46_new = p46_raw[p46_raw.method.isin(["supir", "hypir", "hypir_caption"])].copy()
        p46_new["source"] = "phase4_6_supir_hypir"
        print(f"phase4.6 new: {len(p46_new)} rows", file=sys.stderr)
    else:
        p46_new = pd.DataFrame()

    final = pd.concat([p3, lora_rows, p46_new], ignore_index=True, sort=False)
    out_path = EVAL_DIR / "final_leaderboard.csv"
    final.to_csv(out_path, index=False)
    print(f"\nWrote {len(final)} rows to {out_path}", file=sys.stderr)

    # Summary
    print("\n=== Mean LPIPS by method x ratio ===", file=sys.stderr)
    print(
        final.groupby(["method", "lr_size"]).lpips.mean().unstack("lr_size").round(4).to_string(),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
