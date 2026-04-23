# SD_image_upscaler

A diffusion-based image upscaler built as a 50-hour capability study against
SUPIR (CVPR 2024) and HYPIR. Targets 200×200 → 1000×1000 (5× linear), with 4×
and aspirational 10× variants.

**Status:** in development. See `docs/project_plan.md` for the authoritative
plan and `CLAUDE.md` for working conventions.

## Quickstart

```bash
# One-time setup (requires uv)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Quality gates
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Layout

- `src/upscaler/` — library code (imported as `upscaler`).
- `scripts/` — CLI wrappers over `upscaler` for reproducibility.
- `notebooks/` — thin consumers that import from `upscaler`.
- `tests/` — deterministic, GPU-free, CI-safe.
- `configs/` — YAML configs for training runs.
- `data/`, `outputs/` — gitignored; populated during runs.
- `docs/` — plan, reference material.

## License

MIT. See `LICENSE`.
