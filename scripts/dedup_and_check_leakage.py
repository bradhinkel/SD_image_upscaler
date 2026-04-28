"""Phase 4a quality gate: deduplicate training tiles and check for test-set
leakage via perceptual hashes.

What it does:
1. Computes pHash for every HR tile under data/pairs/ and every HR test image
   in data/test_images/ (60 frozen test images).
2. Within-training: groups tiles by exact pHash collision, flags duplicate
   groups.
3. Cross-set: for each training tile, computes Hamming distance to every test
   tile's pHash. Flags any pair with distance <= --leak-threshold (default 8).
4. Writes outputs/phase4/leakage_report.md with all findings.
5. With --evict, deletes the dupe/leak pairs (both _hr and _lr files).

Re-runnable. By default does NOT delete anything — review the report, then
re-run with --evict if you accept the proposals.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent


def phash(path: Path) -> imagehash.ImageHash:
    return imagehash.phash(Image.open(path).convert("RGB"))


def is_hr_test_image(p: Path) -> bool:
    """True for the 60 HR test images, False for their _100/_200/_250 LR variants."""
    return not any(p.stem.endswith(f"_{s}") for s in (100, 200, 250))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "data" / "pairs")
    parser.add_argument("--test-dir", type=Path, default=REPO_ROOT / "data" / "test_images")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "phase4" / "leakage_report.md"
    )
    parser.add_argument(
        "--leak-threshold",
        type=int,
        default=8,
        help="pHash Hamming distance threshold for flagging cross-set near-dupes",
    )
    parser.add_argument(
        "--evict",
        action="store_true",
        help="Actually delete the duplicate/leak pair files (both _hr and _lr).",
    )
    args = parser.parse_args()

    # Hash test images (HR only).
    test_paths = sorted(p for p in args.test_dir.glob("*.jpg") if is_hr_test_image(p))
    print(f"Hashing {len(test_paths)} HR test images...", file=sys.stderr)
    test_hashes: dict[Path, imagehash.ImageHash] = {
        p: phash(p) for p in tqdm(test_paths, desc="test")
    }

    # Hash training HR tiles.
    pair_paths = sorted(args.pairs_dir.glob("*_hr.jpg"))
    print(f"Hashing {len(pair_paths)} training HR tiles...", file=sys.stderr)
    pair_hashes: dict[Path, imagehash.ImageHash] = {
        p: phash(p) for p in tqdm(pair_paths, desc="pairs")
    }

    # Within-training: exact pHash collisions.
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p, h in pair_hashes.items():
        by_hash[str(h)].append(p)
    dedup_groups = [paths for paths in by_hash.values() if len(paths) > 1]

    # Cross-set leakage: any training tile within --leak-threshold of any test image.
    print("Computing leakage distances...", file=sys.stderr)
    leaks: list[tuple[Path, Path, int]] = []
    for pair_path, ph in tqdm(pair_hashes.items(), desc="leak check"):
        for test_path, th in test_hashes.items():
            d = ph - th
            if d <= args.leak_threshold:
                leaks.append((pair_path, test_path, d))

    # Plan eviction set.
    to_evict_hr: set[Path] = set()
    for grp in dedup_groups:
        # Keep first, evict the rest.
        for p in sorted(grp)[1:]:
            to_evict_hr.add(p)
    for pair_path, _test, _d in leaks:
        to_evict_hr.add(pair_path)

    # Write report.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("# Phase 4a — dedup + test-set leakage report\n\n")
        f.write(f"- Training pairs scanned: **{len(pair_paths)}**\n")
        f.write(f"- HR test images scanned: **{len(test_paths)}**\n")
        f.write("- Dedup criterion: exact pHash match within training\n")
        f.write(f"- Leakage criterion: pHash Hamming distance <= **{args.leak_threshold}**\n\n")
        f.write("## Within-training duplicates\n\n")
        if not dedup_groups:
            f.write("None.\n\n")
        else:
            f.write(f"{len(dedup_groups)} duplicate groups, ")
            f.write(f"{sum(len(g) - 1 for g in dedup_groups)} pairs marked for eviction:\n\n")
            for i, grp in enumerate(sorted(dedup_groups, key=len, reverse=True)[:30]):
                f.write(f"- group {i + 1} ({len(grp)} matches):\n")
                for p in sorted(grp):
                    keep = " (keep)" if p == sorted(grp)[0] else ""
                    f.write(f"  - `{p.relative_to(REPO_ROOT)}`{keep}\n")
            if len(dedup_groups) > 30:
                f.write(f"\n_(showing top 30 of {len(dedup_groups)} groups)_\n")

        f.write("\n## Test-set leakage\n\n")
        if not leaks:
            f.write("None at this threshold.\n\n")
        else:
            f.write(f"{len(leaks)} flagged training-tile / test-image pairs:\n\n")
            f.write("| Training tile | Test image | pHash distance |\n|---|---|---|\n")
            for pair_path, test_path, d in sorted(leaks, key=lambda x: x[2])[:50]:
                f.write(f"| `{pair_path.name}` | `{test_path.name}` | {d} |\n")
            if len(leaks) > 50:
                f.write(f"\n_(showing closest 50 of {len(leaks)} flagged pairs)_\n")

        f.write("\n## Eviction\n\n")
        f.write(f"- Pairs proposed for eviction: **{len(to_evict_hr)}**\n")
        if args.evict:
            evicted = 0
            for hr in to_evict_hr:
                idx = hr.stem.split("_")[0]
                lr = hr.parent / f"{idx}_lr.jpg"
                hr.unlink(missing_ok=True)
                lr.unlink(missing_ok=True)
                evicted += 1
            f.write(f"- Eviction status: **EXECUTED** ({evicted} pairs deleted)\n")
            print(f"\nEvicted {evicted} pairs.", file=sys.stderr)
        else:
            f.write("- Eviction status: **DRY RUN** — re-run with `--evict` to delete.\n")

    print(f"\nReport written to: {args.out}", file=sys.stderr)
    print(
        f"Summary: {len(dedup_groups)} dupe groups, {len(leaks)} leak flags, "
        f"{len(to_evict_hr)} pairs proposed for eviction.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
