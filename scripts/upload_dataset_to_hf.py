"""Push the Phase 4a dataset (data/pairs/ + data/captions.jsonl) to a private
HuggingFace Hub dataset repo.

Reads HF_TOKEN from .env. Creates the repo if it doesn't exist. Uses
`huggingface_hub.upload_folder` which handles delta uploads and large file
counts efficiently.

Usage:
    python -m scripts.upload_dataset_to_hf                      # dry-run
    python -m scripts.upload_dataset_to_hf --execute            # upload for real
    python -m scripts.upload_dataset_to_hf --repo-id user/name --execute
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Slug-style HF dataset repo by default. Can be overridden with --repo-id.
DEFAULT_REPO_ID = "bradhinkel/sd-image-upscaler-pairs"

DATASET_README = """\
# SD Image Upscaler — Training Pairs (Phase 4a)

Private training-pair dataset for the **SD Image Upscaler** capability study.

- **HR tiles:** 512x512 random crops from DIV2K (800 imgs) + Unsplash Lite
  (1800 filtered photos: 600 landscape, 600 cityscape, 600 animals).
- **LR tiles:** 128x128 generated from each HR via a Real-ESRGAN-style
  degradation pipeline (random Gaussian blur, downsample method, additive
  noise, JPEG compression).
- **Captions:** one BLIP-large generated caption per HR tile, in JSONL.
- **Quality gates:** within-training pHash dedup + leakage check vs the
  60-image frozen test set (passed: 5 dupe groups evicted, 0 test-set leaks).

## Layout

Pairs are bundled as a single tarball (HF Hub's per-repo commit rate limit
makes the 15k-file flat upload impractical):

```
pairs.tar           # 916 MB, plain tar (JPEGs don't compress further)
                    # extracts to pairs/NNNNNN_(hr|lr).jpg
captions.jsonl      # {"idx": int, "caption": str, "hr_path": str}
README.md
```

Extract once on the training host:
```bash
huggingface-cli download bradhinkel/sd-image-upscaler-pairs \\
    --repo-type dataset --local-dir data/
tar -xf data/pairs.tar -C data/
```

## Reproduction

See [github.com/bradhinkel/SD_image_upscaler](https://github.com/bradhinkel/SD_image_upscaler)
for the full pipeline. Phase 4a in `CLAUDE.md` enumerates every script that
produced this dataset.

## Source attributions

- DIV2K: research-only license, [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Unsplash Lite Dataset: ML training permitted under the Lite Dataset terms,
  [https://github.com/unsplash/datasets](https://github.com/unsplash/datasets)
- BLIP captions: model `Salesforce/blip-image-captioning-large`

## License

Private repo, internal use only.
"""


def load_hf_token() -> str:
    """HF_TOKEN from process env or .env at repo root."""
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF Hub dataset repo (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "pairs")
    parser.add_argument("--tarball", type=Path, default=REPO_ROOT / "data" / "pairs.tar")
    parser.add_argument("--captions", type=Path, default=REPO_ROOT / "data" / "captions.jsonl")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the dataset repo as public (default: private).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually push (default is a dry-run that prints what would happen).",
    )
    args = parser.parse_args()

    if not args.pairs_dir.is_dir():
        print(f"Pairs dir not found: {args.pairs_dir}", file=sys.stderr)
        return 1
    if not args.captions.is_file():
        print(f"Captions file not found: {args.captions}", file=sys.stderr)
        return 1

    token = load_hf_token()
    if not token:
        print(
            "HF_TOKEN missing. Add `HF_TOKEN=hf_...` to .env (write scope) and rerun.",
            file=sys.stderr,
        )
        return 1

    pair_count = sum(1 for _ in args.pairs_dir.glob("*_hr.jpg")) if args.pairs_dir.is_dir() else 0
    cap_lines = sum(1 for _ in args.captions.open())
    tar_size = args.tarball.stat().st_size if args.tarball.is_file() else 0
    print(f"Repo:     {args.repo_id}", file=sys.stderr)
    print(f"Tarball:  {args.tarball} ({tar_size / 1e9:.2f} GB)", file=sys.stderr)
    print(f"  bundles {pair_count} HR + {pair_count} LR pair files", file=sys.stderr)
    print(f"Captions: {cap_lines} lines", file=sys.stderr)
    print(f"Privacy:  {'public' if args.public else 'private'}", file=sys.stderr)

    if not args.execute:
        print("\nDRY RUN — re-run with --execute to push.", file=sys.stderr)
        return 0

    if not args.tarball.is_file():
        print(
            f"Tarball missing: {args.tarball}\n"
            f"  Build it first: cd {args.tarball.parent} && tar -cf pairs.tar pairs/",
            file=sys.stderr,
        )
        return 1

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create or reuse the dataset repo (idempotent via exist_ok).
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        token=token,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}", file=sys.stderr)

    # Three single-file uploads, three commits — well under HF Hub's
    # 128-commits-per-hour rate limit (we hit that with upload_large_folder
    # because 15k files chunked into many small commits).
    readme_path = args.pairs_dir.parent / "README.md"
    if not readme_path.is_file() or readme_path.read_text() != DATASET_README:
        readme_path.write_text(DATASET_README)

    print("Uploading README.md...", file=sys.stderr)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Phase 4a: dataset card",
    )

    print("Uploading captions.jsonl...", file=sys.stderr)
    api.upload_file(
        path_or_fileobj=str(args.captions),
        path_in_repo="captions.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Phase 4a: captions",
    )

    print(
        f"Uploading {args.tarball.name} (~{args.tarball.stat().st_size / 1e9:.2f} GB) — single file, single commit...",
        file=sys.stderr,
    )
    api.upload_file(
        path_or_fileobj=str(args.tarball),
        path_in_repo="pairs.tar",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Phase 4a: training pairs (tarball)",
    )

    print("\nDone.", file=sys.stderr)
    print(f"https://huggingface.co/datasets/{args.repo_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
