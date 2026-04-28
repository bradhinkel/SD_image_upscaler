"""Caption every HR_512 tile under data/pairs/ with BLIP and write JSONL.

Each line: {"idx": int, "caption": str, "hr_path": str}

Idempotent: skips indices already present in an existing captions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from upscaler.captioning import DEFAULT_MODEL_ID, BLIPCaptioner

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "pairs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "captions.jsonl")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    hr_paths = sorted(args.pairs_dir.glob("*_hr.jpg"))
    if not hr_paths:
        print(f"No *_hr.jpg in {args.pairs_dir}", file=sys.stderr)
        return 1

    # Resume-safe: read existing captions and skip those indices.
    seen: set[int] = set()
    if args.out.is_file():
        with args.out.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(int(json.loads(line)["idx"]))
        print(f"Resuming: {len(seen)} captions already in {args.out}", file=sys.stderr)

    todo = [p for p in hr_paths if int(p.stem.split("_")[0]) not in seen]
    if not todo:
        print(f"All {len(hr_paths)} HR tiles already captioned.", file=sys.stderr)
        return 0
    print(f"Captioning {len(todo)} tiles with {args.model_id}", file=sys.stderr)

    cap = BLIPCaptioner(model_id=args.model_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out_f:
        for i in tqdm(range(0, len(todo), args.batch_size), desc="caption"):
            batch_paths = todo[i : i + args.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            captions = cap.caption_batch(images, max_new_tokens=args.max_new_tokens)
            for path, caption in zip(batch_paths, captions, strict=True):
                idx = int(path.stem.split("_")[0])
                rec = {"idx": idx, "caption": caption, "hr_path": str(path.relative_to(REPO_ROOT))}
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    print(f"\nWrote captions to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
