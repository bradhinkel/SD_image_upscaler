# Diffusion Image Upscaler — Project Plan

**Author:** Brad
**Date:** 2026-04-19
**Effort target:** ~50 hours of focused work, spread across ~3–4 weeks of evenings and weekends. Tracked deliberately so the final writeup can honestly claim the number.
**Demo targets:**
  - **200×200 → 1000×1000 (5× linear)** — headline demo, reliable output that looks crisp on a normal monitor. This is the primary "it works" deliverable.
  - **100×100 → 1000×1000 (10× linear)** — stretch demo, same pipeline pushed to its fidelity cliff. Educational: shows where invention takes over from reconstruction.
**Reference points:** SUPIR (CVPR 2024) and **HYPIR** (2026, SUPIR's improved commercial successor, available via suppixel.ai). The project is framed as *"what can one person build on consumer hardware in 50 hours, compared to a 2024 research SOTA and its current productized descendant?"*
**Repo / package:** GitHub repo `SD_image_upscaler`; Python package `upscaler` (import as `from upscaler.pipeline import UpscalerPipeline`).
**Subject domain:** landscape / nature and architecture / urban photographs. Training data from public datasets; personal photos reserved for the held-out test set.
**Cloud budget:** ~$50 total — ~$14 SD 1.5 LoRA on DO H100, ~$20 SDXL LoRA on DO H100, $5–15 suppixel.ai credits for SUPIR + HYPIR benchmark, ~$5 contingency.
**Final deployment:** existing 4 GB Digital Ocean droplet as the always-on frontend, with inference offloaded to a Modal serverless GPU endpoint that scales to zero when idle.

---

## 0. Working mode — read this first (for Claude Code and Brad)

This project is built **one phase at a time**. The acceptance criteria in each phase are the contract.

For each phase, the workflow is:

1. Claude Code reads the phase section and its acceptance criteria in full.
2. Claude Code does the work, with regular commits.
3. When Claude Code believes the acceptance criteria are met, it **stops**. It does **not** start the next phase.
4. Claude Code writes a short phase-completion summary: what was built, verification output for each acceptance criterion, any plan deviations (with reasons), and what Brad should look at first when reviewing.
5. **Brad reviews the phase, gives feedback, and explicitly approves it before the next phase starts.** Push-back, scope changes, and plan revisions are expected — the plan was shaped iteratively and is intentionally revisable.
6. Only once a phase is accepted does the next one begin.

If the plan is ambiguous, wrong, or would require a meaningful deviation mid-phase, Claude Code **stops and asks** rather than guessing. This is a collaborative project; the plan is not sacred.

Operator-level details — commands, conventions, gotchas — live in `CLAUDE.md`. Secrets live in `.env` (gitignored), modeled on `.env.example`.

---

## 1. Key decisions (and why)

| Decision | Choice | Reason |
|---|---|---|
| Base model | **Start with SD 1.5, expect SDXL to win** | SD 1.5 is the pipeline-rehearsal platform — fits comfortably in 8–10 GB VRAM, fast iteration, cheap training. SDXL is the likely production model based on quality; you'll compare them empirically in Phase 4 and deploy whichever wins. |
| Upscaling backbone | **`stable-diffusion-x4-upscaler`** for the first pass, **ControlNet Tile** for the second pass | `x4-upscaler` is a purpose-built latent upscaler with a natural 4× ratio. For 5× total we push it slightly past native; for 10× we lean heavily on stage B. |
| Training | **LoRA with captioned training data** | LoRA fits your VRAM and hits the 50-hour target. Captions (generated per-image with BLIP-2) meaningfully improve quality — SUPIR uses the same trick with LLaVA. |
| Training data | **DIV2K + LSDIR + optional Flickr2K — public only** | Your personal photos are the *held-out test set*, never seen by the model. Public datasets give the volume + variety training needs and avoid licensing complications if you publish LoRA weights. |
| Deliverable | **Python package + notebooks + SUPIR-comparison writeup** | Core logic in `src/upscaler/` with tests. Notebooks are the visual lab bench importing from the package. Final README is the comparison-study narrative, not just install docs. |
| Environment | **WSL2 + Ubuntu 22.04 + CUDA 12.x + PyTorch 2.x** | Cleanest NVIDIA stack on Windows. Avoids native-Windows `bitsandbytes`/`xformers` pain. |
| Library | **`diffusers` + `peft` + `accelerate` + `transformers` (for BLIP-2)** | The de-facto stack. `peft` has the cleanest LoRA implementation; BLIP-2 handles dataset captioning. |
| Cloud GPU | **DO GPU droplet (H100, $3.39/hr), used twice** | ~$14 SD 1.5 LoRA run (~4 hr) + ~$20 SDXL LoRA run (~6 hr). Single-provider consolidation with your existing DO account. H100 is overkill for SD 1.5 — we pay for simplicity. SUPIR + HYPIR benchmark is no longer cloud GPU; runs manually via suppixel.ai. |
| RAM spillover | **Used deliberately for overnight jobs** | Your card can swap to system RAM. Expect 5–30× slowdown but it unlocks larger configs without cloud cost. Useful for SDXL *inference* at home during development. |

## 2. Pipeline architecture

```
input (100x100 stretch / 200x200 headline)
  │
  ▼
[stage A] stable-diffusion-x4-upscaler
          (latent, 25 steps, noise_level=20)
          conditioned on caption text + LR image
  │
  ▼
intermediate (400x400 / 800x800)
  │
  ▼
[stage B] bicubic upsample to 1000x1000, then
          img2img + ControlNet Tile on SD1.5/SDXL
          (denoise 0.20–0.40, tiled if needed)
          [+ trained LoRA]
          [+ caption prompt]
  │
  ▼
output 1000x1000
```

Stage A gets you native 4× with a model purpose-built for it. Stage B handles the remaining ratio with tile-based control so it doesn't reshape scene structure — just adds texture. The 200→1000 path keeps stage B light (1.25× further scale). The 100→1000 path leans on stage B for a much bigger lift, and that's where the fidelity cliff appears.

## 3. Repository structure

Package-first layout. Notebooks import from `upscaler/`; they don't own logic.

```
SD_image_upscaler/
├── pyproject.toml              # uv-managed, pinned deps, pip-installable
├── README.md                   # SUPIR-comparison writeup (see §8) + install/run
├── docker-compose.yml          # one-command local self-host (frontend + GPU worker)
├── .env.example                # HF token, Modal token, etc.
│
├── src/upscaler/
│   ├── __init__.py
│   ├── pipeline.py             # UpscalerPipeline: load models, run two-stage inference
│   ├── tiling.py               # tile/merge logic for high-resolution outputs
│   ├── degradations.py         # Real-ESRGAN-style LR degradation for training pairs
│   ├── dataset.py              # torch Dataset for (LR, HR, caption) triples
│   ├── captioning.py           # BLIP-2 wrapper: caption_image(img) -> str
│   ├── eval_metrics.py         # LPIPS, DISTS, PSNR, SSIM wrappers
│   ├── baselines.py            # bicubic, Lanczos, Real-ESRGAN wrappers
│   ├── lora_train.py           # training loop entrypoint (local + cloud)
│   └── modal_app.py            # Modal serverless handler (Phase 6)
│
├── notebooks/                  # visual lab bench — import from upscaler
│   ├── 01_baseline.ipynb           # bicubic / Lanczos / x4-upscaler / Real-ESRGAN
│   ├── 02_two_stage.ipynb          # full pipeline, 4×/5×/10× parameter sweep
│   ├── 025_vae_swap.ipynb          # VAE comparison experiment
│   ├── 03_eval.ipynb               # leaderboard + SUPIR comparison
│   ├── 035_tiny_diffusion.ipynb    # from-scratch DDPM (logic lives here deliberately)
│   ├── 04_prompt_ladder.ipynb      # 6-level prompting experiment
│   └── 05_lora_results.ipynb       # before/after each LoRA variant
│
├── frontend/                   # Phase 5/6 Gradio app
│   ├── main.py                 # FastAPI + Gradio + SQLite queue
│   ├── Dockerfile
│   └── requirements.txt
│
├── configs/                    # training run configs
│   ├── rehearsal.yaml          # rank 8, 128 crops, 1000 steps
│   ├── sd15_main.yaml          # rank 16–32, 512 crops, 8–10k steps
│   └── sdxl_main.yaml          # same shape, SDXL base
│
├── tests/                      # pytest; runs in CI; no GPU needed
│   ├── test_tiling.py
│   ├── test_degradations.py
│   ├── test_eval_metrics.py
│   └── test_pipeline_smoke.py
│
├── scripts/                    # standalone CLI utilities
│   ├── build_training_pairs.py
│   ├── caption_dataset.py      # BLIP-2 pre-captions all training images
│   ├── run_inference.py        # python -m scripts.run_inference in.jpg --scale 5
│   └── benchmark_pipeline.py
│
├── data/                       # gitignored; test images + datasets
└── outputs/                    # gitignored; sample outputs, LoRA checkpoints, eval CSVs
```

**Rules of thumb:**
- If a block of code would be tested or reused, it belongs in `src/upscaler/`.
- Notebooks contain *orchestration and display* — load, sweep, render, interpret.
- `scripts/` are thin CLI wrappers over `upscaler/` for reproducibility.
- Tests cover deterministic logic (tiling math, degradations, metric ranges). No GPU required to run the suite.

## 4. Phases

Estimated hours against the 50-hour target are shown in parentheses.

### Phase 0 — Environment and package scaffold (~3 hours)
- Install WSL2 Ubuntu 22.04, NVIDIA driver on Windows host, CUDA toolkit inside WSL.
- Scaffold the repo per the structure in §3: `pyproject.toml`, empty `src/upscaler/`, `tests/`, `notebooks/`, `configs/`.
- Use **uv** for dependency management. Pin: `torch`, `diffusers`, `transformers`, `accelerate`, `peft`, `xformers`, `safetensors`, `controlnet_aux`, `lpips`, `pillow`, `numpy`, `jupyter`, `pytest`, `ruff`.
- Pre-commit hooks: `ruff` format + lint, `pytest` smoke.
- `uv pip install -e .` so notebooks and scripts pick up edits instantly.
- Smoke test: single-cell notebook imports `upscaler` and runs SD 1.5 text-to-image at 512×512 to confirm CUDA + xformers work.
- Commit `CLAUDE.md` (operator manual) and `.env.example` at repo root.
- **Deliverable:** repo on GitHub as `SD_image_upscaler`, CI running `ruff check` + `pytest`.

**Acceptance criteria:**
- [ ] Repo pushed to GitHub as `SD_image_upscaler`.
- [ ] `uv pip install -e .` and `uv pip install -e ".[dev]"` complete cleanly.
- [ ] `nvidia-smi` in WSL shows the GPU; `python -c "import torch; assert torch.cuda.is_available()"` exits 0.
- [ ] `uv run ruff check .` and `uv run pytest` both exit 0.
- [ ] GitHub Actions CI workflow exists and has succeeded on at least one push.
- [ ] `CLAUDE.md` and `.env.example` committed at repo root.
- [ ] A smoke notebook runs end-to-end, imports `upscaler`, and generates one SD 1.5 text-to-image at 512×512.
- [ ] Pre-commit hooks installed; `pre-commit run --all-files` passes.

### Phase 1 — Baseline upscaler (~5 hours)
**Package work (`src/upscaler/pipeline.py`, `src/upscaler/baselines.py`)**
- Define `UpscalerPipeline` class with `.load()` and `.upscale_x4(image, prompt, noise_level, steps) -> Image`.
- Lazy-load models on first call; `.close()` releases them.
- Baselines: `bicubic(img, scale)`, `lanczos(img, scale)`, `realesrgan(img, scale)`.

**Tests (`tests/test_pipeline_smoke.py`)**
- Mock the diffusers pipeline; assert `.upscale_x4()` returns an image with expected dimensions. No GPU or model weights required.

**Test set — the frozen benchmark**

Brad curates a 60-image test set, all cropped to **1000×1000 squares** (the project's canonical output size), split as:

- **Traditional (30):** 10 `landscape`, 10 `animals`, 10 `cityscape`. Standard, well-behaved photography — clean baseline conditions.
- **Hard (30):** a cross-section of diffusion-upscaling failure modes, tagged with one or more of the six challenge categories: `text` (readable signage), `fine_architecture` (ornate / dense window grids / repeating patterns), `hf_texture` (forest canopy, grass, distant foliage), `reflection` (glass facades, water, wet surfaces), `noise` (high-ISO or astro shots with real sensor grain), `night` (low-light urban or astro scenes).

Per test image, three LR variants are generated:

| LR size | Scale to 1000×1000 | Purpose |
|---|---|---|
| `{name}_250.jpg` | 4× | Native x4-upscaler stage A only |
| `{name}_200.jpg` | 5× | Headline demo (stage A + light stage B) |
| `{name}_100.jpg` | 10× | Stretch demo (stage A + heavy stage B) |

Brad creates `data/test_images/metadata.json` tagging each image:

```json
{
  "img01_coast_vista.jpg":     {"category": "traditional", "subcategory": "landscape"},
  "img31_storefront.jpg":      {"category": "hard", "challenges": ["text"]},
  "img32_cathedral.jpg":       {"category": "hard", "challenges": ["fine_architecture"]},
  "img33_night_skyline.jpg":   {"category": "hard", "challenges": ["night", "text", "reflection"]}
}
```

Metadata drives the evaluation harness: notebooks slice leaderboards by category and by individual challenge tag, so the final writeup can say "here's performance on traditional photography" alongside "here's where it breaks on text / ornate architecture / HF texture."

**Notebook (`notebooks/01_baseline.ipynb`)**
- Import `UpscalerPipeline` + baseline helpers.
- Load `data/test_images/metadata.json`; slice the test set by category and challenge tag.
- Run bicubic, Lanczos, `stable-diffusion-x4-upscaler`, Real-ESRGAN on each image at each LR variant.
- Render **per-slice contact sheets** (one per category, one per challenge) at 100% zoom crops. A single all-60-images contact sheet would be unusable.

**Deliverables**
- Working `UpscalerPipeline` with smoke tests in CI.
- Per-slice contact sheets in `outputs/phase1/`.
- The frozen 60-image test set + metadata file — used for every subsequent comparison throughout the project.

**Acceptance criteria:**
- [ ] `src/upscaler/pipeline.py` defines `UpscalerPipeline` with `.load()`, `.close()`, `.upscale_x4(image, prompt, noise_level, steps) -> Image`.
- [ ] `src/upscaler/baselines.py` exposes `bicubic`, `lanczos`, `realesrgan`.
- [ ] `src/upscaler/testset.py` (or equivalent) exposes a helper to load the test-set metadata and yield images by slice.
- [ ] `tests/test_pipeline_smoke.py` has ≥3 test cases and passes on CPU with a mocked diffusers pipeline.
- [ ] `tests/test_testset.py` covers metadata loading and slice queries.
- [ ] `data/test_images/` contains **60 images** — 30 traditional (10 landscape, 10 animals, 10 cityscape) + 30 hard (cross-section of text, fine_architecture, hf_texture, reflection, noise, night) — each a 1000×1000 square with `_100.jpg`, `_200.jpg`, `_250.jpg` LR variants. Set is **frozen** — no additions or removals after this phase.
- [ ] `data/test_images/metadata.json` tags each image with category and, for hard images, one or more challenge tags.
- [ ] `notebooks/01_baseline.ipynb` runs via "Restart & Run All" with no errors.
- [ ] `outputs/phase1/` contains per-slice contact sheets showing bicubic / Lanczos / x4-upscaler / Real-ESRGAN side-by-side.

### Phase 2 — Two-stage pipeline at 4× / 5× / 10× (~6 hours)
**Package work**
- Extend `UpscalerPipeline` with `.upscale_two_stage(image, target_size, denoise, steps, cn_weight, prompt) -> Image`.
- New module `src/upscaler/tiling.py`: pure-function tiling — overlapping tiles with feathered blend weights, per-tile callable, stitch. Deterministic, GPU-free, heavily testable.
- `UpscalerPipeline` uses `tiling.py` for tiled VAE decode when output exceeds VRAM.

**Tests (`tests/test_tiling.py`)**
- Tile grid correctness for known input dims.
- Blend weights sum to 1 in overlap regions.
- Round-trip `split → identity → merge` returns input byte-for-byte.
- This is the highest-leverage test in the project — tile seams are the most common visual bug.

**Notebook (`notebooks/02_two_stage.ipynb`)**
- Parameter sweep: denoise strength (0.20 / 0.30 / 0.40 / 0.55), steps (20/30/50), ControlNet weight (0.5 / 0.8 / 1.0), generic prompt.
- Run **each test image at three output ratios**: 4× (native stage A only), 5× (200→1000, headline), 10× (100→1000, stretch).
- Render per-image grids showing all three ratios × several denoise settings.

**Deliverables**
- `upscale_two_stage` public method with well-tested tiling.
- Three parallel fidelity-vs-invention grids — the visual headline of the project.

**Acceptance criteria:**
- [ ] `UpscalerPipeline.upscale_two_stage(image, target_size, denoise, steps, cn_weight, prompt) -> Image` implemented.
- [ ] `src/upscaler/tiling.py` with pure-function tile/merge logic.
- [ ] `tests/test_tiling.py` has ≥5 tests, all passing: tile grid, blend weights sum to 1 in overlap, round-trip identity returns input byte-for-byte, edge-tile handling, non-square input.
- [ ] `notebooks/02_two_stage.ipynb` produces, per test image, a grid of 4× / 5× / 10× × several denoise settings.
- [ ] Output grids in `outputs/phase2/`.
- [ ] At least one grid clearly annotates the 5×→10× fidelity cliff.

### Phase 2.5 — VAE swap experiment (~2 hours)
**Notebook (`notebooks/025_vae_swap.ipynb`)**
- Same test set, same two-stage pipeline, vary the VAE only:
  - SD 1.5 default.
  - `madebyollin/sdxl-vae-fp16-fix` — SDXL VAE backported to SD 1.5.
  - `stabilityai/sd-vae-ft-mse` — official SD 1.5 VAE improvement.
  - `madebyollin/taesd` — tiny VAE for speed reference.
- Render zoomed crops focusing on the VAE's failure mode: fine repeating texture (foliage, fabric, brickwork).
- Pick a VAE to carry forward into Phase 4 training.

**Deliverable:** a one-notebook empirical answer to whether the default VAE is the bottleneck.

**Acceptance criteria:**
- [ ] `notebooks/025_vae_swap.ipynb` runs end-to-end, covering all four VAEs on the full test set.
- [ ] `outputs/phase2_5/vae_comparison.csv` with per-image LPIPS per VAE.
- [ ] Notebook ends with a markdown cell stating the chosen VAE for Phase 4 + one-sentence justification.

### Phase 3 — Evaluation harness + preliminary leaderboard (~4 hours)
Phase 3 establishes the evaluation machinery and produces the first leaderboard — covering baselines and our pre-LoRA pipeline across the full 60-image test set. The SUPIR/HYPIR comparison is deferred to Phase 4.6, after LoRA training, so the comparison-image selection can be informed by how our own pipeline actually performs.

**Package work (`src/upscaler/eval_metrics.py`)**
- Thin tested wrappers: `lpips(pred, target)`, `dists(pred, target)`, `psnr(pred, target)`, `ssim(pred, target)`, each returning a float.
- `evaluate_method(method_name, method_fn, test_pairs) -> pd.DataFrame`.
- `scripts/benchmark_pipeline.py` — CLI that writes a leaderboard CSV to `outputs/eval/`.

**Tests (`tests/test_eval_metrics.py`)**
- Known pairs: identical → LPIPS ≈ 0; random → LPIPS > threshold.
- Catches higher-is-better vs lower-is-better bugs.

**Notebook (`notebooks/03_eval.ipynb`)**
- Uses HR originals as ground truth, downsampled to 100×100 / 200×200 / 250×250 as inputs.
- Evaluates **across the full 60-image set**: bicubic, Lanczos, Real-ESRGAN, our two-stage pipeline (pre-LoRA).
- Renders leaderboard bar chart at each ratio (4× / 5× / 10×), LPIPS + DISTS + PSNR + SSIM (latter two annotated "misleading").
- Slices results by category (traditional vs hard) and by challenge tag.
- Records Brad's subjective human ranking on 3–5 images next to the metric ranking; shows correlation.
- Visual gallery: same test image, every method, side-by-side.

**Deliverable:** preliminary leaderboard CSV + notebook. This notebook becomes the dashboard that's re-run after Phase 4 (with LoRA applied) and again after Phase 4.6 (with SUPIR + HYPIR added).

**Acceptance criteria:**
- [ ] `src/upscaler/eval_metrics.py` exposes `lpips`, `dists`, `psnr`, `ssim`, `evaluate_method`.
- [ ] `tests/test_eval_metrics.py` covers both "identical input" and "random input" edge cases per metric; all pass.
- [ ] `scripts/benchmark_pipeline.py` runnable; produces a CSV.
- [ ] `outputs/eval/leaderboard_phase3.csv` has rows for bicubic, Lanczos, Real-ESRGAN, two-stage (no LoRA) across the **full 60-image set** × each ratio — with LPIPS + DISTS + PSNR + SSIM.
- [ ] `notebooks/03_eval.ipynb` renders leaderboard bar charts sliced by (a) overall, (b) traditional vs hard, (c) per challenge tag. Plus gallery and metric-vs-human correlation.

### Phase 3.5 — Warm-up: tiny diffusion model from scratch (~6 hours, local only)
**This phase deliberately breaks the package-first rule.** The point is to see every piece of the diffusion loop next to markdown that explains it.

**Notebook (`notebooks/035_tiny_diffusion.ipynb`)**
- Goal: 64×64 → 256×256 on a small dataset (~10k-image LSDIR subset).
- Architecture: ~30M-parameter U-Net in pixel space (no VAE), LR image concatenated to noisy HR input as extra channels.
- Implement inline: forward diffusion (`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`), MSE loss, DDIM reverse sampler. Markdown cells explain each block.
- Train ~5k steps in 1–3 hours at batch 16, 64-pixel crops. Fits in pure VRAM.

**Deliverable:** from-scratch upscaler that makes Phases 1–4 feel transparent. Output quality is expected to be poor — that's not what this phase is for.

**Acceptance criteria:**
- [ ] `notebooks/035_tiny_diffusion.ipynb` runs end-to-end on local GPU.
- [ ] Training completes 5k steps in <4 hours; loss decreases monotonically after early warmup.
- [ ] ≥8 markdown cells explaining concepts (forward process, noise schedule, loss, reverse sampling, etc.).
- [ ] Final model produces recognizable (if low-quality) 64×64 → 256×256 outputs on ≥3 sample inputs.

### Phase 4 — Captioning + LoRA training (~15 hours — the project's center of gravity)

**4a. Dataset construction + captioning**

*Package work*
- `src/upscaler/degradations.py`: `degrade_image(hr, seed) -> lr` with blur kernel + bicubic downsample + JPEG quality 60–85 + optional noise.
- `src/upscaler/captioning.py`: BLIP-2 wrapper, `caption_image(img) -> str` (~500M params, fits alongside SD 1.5).
- `src/upscaler/dataset.py`: `UpscaleDataset` yields (LR, HR, caption) triples with seeded reproducibility.
- `scripts/build_training_pairs.py`: crops HR images into 512×512 patches, stores `(LR_128, HR_512)` pairs.
- `scripts/caption_dataset.py`: one-time pass, captions every HR patch with BLIP-2 and caches to JSONL. ~30 min on your GPU for 2000 images.

*Tests (`tests/test_degradations.py`)*
- Output dimensions correct per scale; JPEG reduces file entropy; degraded ≠ original.

*Data*
- Download **DIV2K** (800 HR, permissive) + **LSDIR** (~86k, use a ~2000-image subset filtered for landscape/architecture) + optional **Flickr2K** (~2650).
- Crop to 512×512 patches. Target: ~2000 final pairs.
- Caption each with BLIP-2 (`scripts/caption_dataset.py`). Spot-check 20 captions for quality.
- Back up pairs + captions to S3 or DO Spaces for the cloud run.

**4b. Local pipeline rehearsal (~2 hours)**
- `src/upscaler/lora_train.py` — training loop using `UpscaleDataset`, `peft` LoRA setup, local logging to `outputs/runs/`.
- Launch: `python -m upscaler.lora_train --config configs/rehearsal.yaml`. Rank 8, 128×128 crops, 1000 steps, batch 1.
- Fits in pure VRAM, runs in 1–2 hours.
- Purpose: shake out bugs in dataset, training loop, captioning integration, checkpointing, logging *before* paying cloud rates.
- Confirm loss decreases and intermediate samples look reasonable.

**4c. SD 1.5 LoRA on DO GPU droplet (~$14, ~4 hours on H100)**
- Brad provisions a DO H100 GPU droplet ($3.39/hr), chooses a region with GPU availability.
- Git-clone repo, `uv pip install -e .`, pull dataset + captions from DO Spaces (same-region transfer is free + fast).
- `python -m upscaler.lora_train --config configs/sd15_main.yaml`. Rank 16–32, alpha = rank. Target: `q_proj`, `k_proj`, `v_proj`, `out_proj` in U-Net cross-attention. Batch 4–8, fp16. 8–10k steps.
- Captions used as text conditioning during training.
- Checkpoints every 500 steps to DO Spaces. Pull final LoRA to local.
- Base: `stabilityai/stable-diffusion-x4-upscaler`. Training code handles the LR-conditioned input channel concatenation.
- **Brad destroys the droplet when done** — `terminate`, not `stop`. Idle H100 billing is real.

**4d. SDXL LoRA on DO GPU droplet (~$20, ~6 hours on H100)**
- Same setup, same H100 droplet class, different config (`configs/sdxl_main.yaml`).
- Targets SDXL img2img + SDXL ControlNet Tile.
- **This is likely the production LoRA**, based on your read of where quality lives. Phase 6 deploys whichever variant wins Phase 4e.
- Consider running 4c and 4d back-to-back on the same droplet instance to save spin-up overhead.

**4e. Evaluation (~2 hours)**
- Re-run `notebooks/03_eval.ipynb` across the **full 60-image set** with each LoRA applied via `pipeline.load_lora(path)`. Output: `outputs/eval/leaderboard_phase4.csv` — now includes rows for SD 1.5+LoRA and SDXL+LoRA alongside the Phase 3 baselines.
- Visual grids in `notebooks/05_lora_results.ipynb`: pre-LoRA vs. SD 1.5+LoRA vs. SDXL+LoRA for each test image.
- Per-image analysis: flag which images each variant wins on, loses on, or surprises on. This output directly feeds Phase 4.6's subset selection.
- Brad makes the **SD 1.5 vs SDXL deployment judgment call** based on this leaderboard.
- **Deliverable:** updated full-set leaderboard, before/after grids, a judgment call on which LoRA goes into deployment, and a per-image win/loss/surprise table that informs Phase 4.6.

**Acceptance criteria for Phase 4 (all sub-phases):**
- [ ] `src/upscaler/degradations.py`, `captioning.py`, `dataset.py` implemented.
- [ ] `tests/test_degradations.py` has ≥4 test cases, all passing.
- [ ] `data/pairs/` contains ≥2000 (LR_128, HR_512) pairs.
- [ ] `data/captions.jsonl` has one caption per pair; 20 random captions spot-checked and look reasonable.
- [ ] Dataset mirrored to S3 / DO Spaces.
- [ ] `src/upscaler/lora_train.py` runs the rehearsal config locally to completion in <3 hours with decreasing loss.
- [ ] SD 1.5 LoRA run on DO H100: spend <$20; `outputs/loras/sd15_main.safetensors` saved and pushed to HF Hub; droplet terminated.
- [ ] SDXL LoRA run on DO H100: spend <$25; `outputs/loras/sdxl_main.safetensors` saved and pushed to HF Hub; droplet terminated.
- [ ] `outputs/eval/leaderboard_phase4.csv` includes both LoRA rows alongside Phase 3 entries.
- [ ] `notebooks/05_lora_results.ipynb` shows before/after grids per test image and ends with a one-paragraph decision on the deployment LoRA.

### Phase 4.5 — Prompting ladder experiment (~4 hours)
**Notebook (`notebooks/04_prompt_ladder.ipynb`)**

Six prompt sources, applied to the full test set through the winning LoRA pipeline:

| Level | Source | Example |
|---|---|---|
| 0 | None | `""` |
| 1 | Generic quality | `"sharp, high detail, professional photograph"` |
| 2 | Hand-written domain | `"landscape photograph of rocky mountains with clear sky"` |
| 3 | Templated | `"a {subject} photograph, sharp focus, high detail"` (subject inferred) |
| 4 | VLM auto-caption | BLIP-2 caption of the LR input |
| 5 | Hand-authored creative | wildly different prompts on one image: "oil painting," "photograph at dusk," "architectural rendering," etc. |

**Per-image grid:** all six levels, fixed denoise, side-by-side. Annotate where prompting helps vs. hurts.

**Hero demonstration:** pick two representative images (one landscape, one architecture), run five intentionally divergent creative prompts per image, show the vastly different outputs — this is the SUPIR-style "same LR, different creative direction" showcase.

**Deliverables**
- Grid notebook.
- An empirical answer to how much prompting matters for upscaling quality.
- Real justification for the prompt box in the Gradio UI.

**Acceptance criteria:**
- [ ] `notebooks/04_prompt_ladder.ipynb` runs all six levels × full test set.
- [ ] `outputs/phase4_5/prompt_comparison.csv` has per-image, per-prompt-level LPIPS values.
- [ ] Hero "same LR, five wildly different prompts" demo rendered for ≥2 images.
- [ ] Notebook ends with a markdown summary of where prompting matters most and least.

### Phase 4.6 — SUPIR / HYPIR benchmark (~3 hours Claude Code + Brad's manual runs)
The reference-SOTA comparison happens here, after Phase 4e has surfaced where our pipeline wins, fails, and surprises. Subset selection is **data-driven** based on the Phase 4e leaderboard, not category quotas — this is what lets the final writeup tell a specific story ("here's the image where SDXL+LoRA beat bicubic decisively, and here's how SUPIR handled the same image").

**Subset selection (Brad's call, informed by Phase 4e output)**

Brad picks 12 comparison images and records the selection + rationale in `outputs/supir/subset.json`. Suggested rubric (adjust as the data dictates):

- **3 "clear wins"** — images where your LoRA pipeline's LPIPS is close to the best non-SUPIR method. Validates competitive quality on images that play to our strengths.
- **3 "clear losses"** — images where your pipeline struggles (worst LPIPS, worst human rank, visible artifacts). Reveals what SUPIR's extra machinery buys on the hard cases.
- **3 "surprises"** — images where SD 1.5+LoRA and SDXL+LoRA disagree sharply, or where LPIPS disagrees with your human ranking. Interesting failure modes worth a direct comparison.
- **3 "coverage"** — one landscape, one cityscape, one hard-challenge image selected purely to keep the subset from being all outliers.

**SUPIR + HYPIR runs (Brad's manual task, via suppixel.ai workbench)**
- 12 images × 3 ratios × 2 models = 72 renders.
- Outputs downloaded with a consistent naming scheme to `outputs/supir/` and `outputs/hypir/`.
- Brad records total credits / $ spent.

**Claude Code's work**
- Extend `notebooks/03_eval.ipynb` (or create `notebooks/046_supir_benchmark.ipynb`) to consume the SUPIR/HYPIR outputs, score them with the same metric wrappers, and fold them into the leaderboard as **subset rows** alongside the full-set rows from Phase 4e.
- Render a head-to-head section: our winning LoRA pipeline vs. SUPIR vs. HYPIR on the 12 subset images, with annotations on the per-image rationale ("this was a clear-win image," "this was a surprise," etc.).

**Deliverable:** final leaderboard CSV (`outputs/eval/leaderboard_phase4_6.csv`) with the SUPIR/HYPIR subset rows merged in, plus a head-to-head analysis notebook that becomes the comparative centerpiece of the writeup.

**Acceptance criteria:**
- [ ] `outputs/supir/subset.json` documents the 12-image selection with a rationale for each image (clear win / clear loss / surprise / coverage).
- [ ] **Brad has run** SUPIR + HYPIR through suppixel.ai on the 12 subset images × 3 ratios; all 72 renders saved to `outputs/supir/` and `outputs/hypir/` with consistent naming.
- [ ] Credits / $ spent on suppixel.ai recorded (in the notebook markdown or a `cost.md`).
- [ ] `outputs/eval/leaderboard_phase4_6.csv` merges the 12-image SUPIR + HYPIR rows with the full-set Phase 4 leaderboard.
- [ ] Head-to-head notebook renders 12-image side-by-side grids (our winning LoRA vs. SUPIR vs. HYPIR) at each of the 3 ratios.
- [ ] Notebook ends with a markdown summary: where we're competitive, where the SOTA gap is largest, and one hypothesis per "clear loss" image for *why* SUPIR handles it better.

### Phase 5 — Application: local Gradio app (~4 hours)
**Package work (`frontend/main.py`)**
- FastAPI app with Gradio mounted at `/`.
- Imports `UpscalerPipeline` from `upscaler/`. Thin UI wrapper — no pipeline logic here.
- Inputs: image upload, target-ratio dropdown (4× / 5× / 10×), denoise slider, steps slider, prompt textbox (default: auto-captioned), LoRA on/off toggle, seed.
- Outputs: upscaled image + per-stage intermediates (LR → stage A → stage B final).
- Works locally at `localhost:7860`.

**Docker (`frontend/Dockerfile` + root `docker-compose.yml`)**
- `docker compose up` on any NVIDIA+Docker machine runs the full demo. **Primary "someone can reproduce the project" path** — the model you run on your own workstation uses this same setup.
- One service (`frontend`) with the pipeline in-process, since local dev has a GPU.

**Deliverables**
- Usable local demo app.
- `docker-compose.yml` for self-hosting.
- Drop-in starting point for Phase 6 — the pipeline call site swaps to a Modal client.

**Acceptance criteria:**
- [ ] `uv run python frontend/main.py` starts Gradio at `localhost:7860`.
- [ ] UI exposes image upload, ratio dropdown (4×/5×/10×), denoise slider, steps slider, prompt textbox, LoRA toggle, seed input.
- [ ] Output includes LR input + stage A intermediate + stage B final.
- [ ] `frontend/Dockerfile` builds clean.
- [ ] `docker-compose.yml` at repo root; `docker compose up` brings up the full app on a local NVIDIA+Docker host, tested end-to-end with a real upscale.
- [ ] README has a "Self-host via Docker" section.

### Phase 6 — Deployment: DO droplet frontend + Modal GPU backend (~6 hours)
The full pipeline at 100→1024 (or 200→1024) needs a real GPU. Hosting a GPU 24/7 for an occasional-use demo is wasteful. Split the system.

**6a. Modal GPU backend (`src/upscaler/modal_app.py`)**
- `modal_app.py` imports `UpscalerPipeline` from the same package. Handler is ~40 lines — deploy wrapper, not a rewrite.
- Decorate: `@app.function(gpu="A10G", image=image, volumes={"/cache": volume}, container_idle_timeout=120)`.
- **Base model is parameterized** — same handler code serves SD 1.5 or SDXL depending on config. GPU choice tracks base model: A10G for SD 1.5, L4 / L40S for SDXL.
- Input: base64 source image, prompt, target ratio, denoise, seed. Output: base64 upscaled image + timing info.
- Container image: `Image.debian_slim().pip_install_from_pyproject("pyproject.toml")` — reuses local dependency pins.
- **Model weight caching** via Modal Volumes at `/cache`. First deploy writes ~4–8 GB. Subsequent cold starts read from volume in seconds.
- LoRA weights uploaded to HF Hub, pulled into the volume on first boot.
- Expected cold start: 20–40 sec with cache. Warm inference: 5–15 sec (SD 1.5) or 10–25 sec (SDXL).
- Cost: ~$5–10/mo at 10 req/day, ~$0 idle. Modal's $30/mo free credit covers it easily.

**6b. Frontend droplet**
- Reuses your existing **4 GB DO droplet** — more than enough.
- Same `frontend/main.py` from Phase 5, with the pipeline call site replaced by a Modal client call (~20 lines changed).
- Add: SQLite job queue, per-IP rate limiter (3 req/IP/hour), hard daily cap, 24h result cache.
- Deploy via `docker compose` over SSH.

**6c. Job flow and UX**
- Upload → return `job_id` immediately.
- Browser polls `/jobs/<id>/status` every 1–2 sec. States: `queued | warming_gpu | inferring | done | failed`.
- Progress UI: "Warming up GPU (first request can take up to a minute)…" during cold start, then "Upscaling…" with estimated time.
- Done includes a shareable URL to the cached output.

**6d. Keep-warm (deferred)**
- Not implemented initially. Cold starts are survivable given low expected traffic.
- If it becomes a problem, a 10-min cron ping during peak hours costs ~$2–5/mo.

**6e. Security and cost guardrails**
- GPU API key in `.env` on the droplet, never to browser.
- Rate limit on the droplet, not the backend.
- Hard daily request cap to kill viral-moment cost spikes.
- SSL via Let's Encrypt, subdomain under a domain you own.

**6f. Deliverables**
- Public URL with the full 200→1000 (and 100→1000) demo at native pipeline quality.
- `modal_app.py` + `frontend/Dockerfile` + `docker-compose.yml` in the repo.
- README reproducing paths: **(a)** `docker compose up` locally with NVIDIA+Docker, or **(b)** set `MODAL_TOKEN` and point at your own Modal deployment.
- Measured cold-start times, warm-inference times, and actual month-to-month cost in the writeup.
- Realistic total: **$10–25/mo** (droplet + Modal).

**Acceptance criteria:**
- [ ] `modal deploy src/upscaler/modal_app.py` succeeds.
- [ ] Cold start measured <90 sec once the Modal Volume cache is populated; warm inference <20 sec.
- [ ] `frontend/main.py` supports `UPSCALER_MODE=local|modal` and both paths work.
- [ ] Frontend deployed to the 4 GB DO droplet via `docker compose`.
- [ ] Live URL reachable over HTTPS with a valid Let's Encrypt cert.
- [ ] Rate limiter enforces 3 req/IP/hour and the daily cap; both exercised with intentional overruns.
- [ ] README has both reproduction paths (local Docker and Modal).
- [ ] Actual one-week cost measured and reported in the writeup.

### Phase 7 — Writeup + final polish (~4 hours)
The 50-hour project's final artifact is the README as a comparison writeup (see §8). This phase pulls the threads together:

- Final leaderboard CSV — every method, every ratio.
- Hero images from `02_two_stage.ipynb`, `04_prompt_ladder.ipynb`, `05_lora_results.ipynb`.
- Total cost tally (cloud + Modal + droplet).
- Hour-tracking total — the honest "it took X hours" number.
- "What I'd do differently" / "What surprised me" section.

**Acceptance criteria:**
- [ ] README restructured per §8 outline.
- [ ] `outputs/eval/final_leaderboard.csv` contains every method across every target ratio.
- [ ] Hero images embedded inline in the README.
- [ ] Hour log + cost log sections filled with real numbers.
- [ ] ≥3 entries under "what I'd do differently" / "what surprised me."
- [ ] Proposed repo subtitle set on GitHub and in the README.

### Stretch goals (out of the 50-hour budget — pick zero or one)
- **Second LoRA** on the stage B refinement pass, trained as a style adapter. Shows two ways of using LoRA in one pipeline.
- **Small custom ControlNet Tile** trained on cloud (~$10 remaining budget).
- **Restoration-guided sampling** — implement SUPIR's Section 3.2 technique (fidelity-quality tunable parameter) as a bolt-on to the two-stage pipeline.
- **Print extension (separate project):** extending the pipeline beyond 1000×1000 to the 60+ MP print target requires smarter cross-tile coherence (MultiDiffusion-style attention), memory streaming, and print-workflow concerns (soft-proofing, color spaces). Explicitly scoped out of this project; noted as follow-on work.

## 5. Risks and things I want to revisit

- **10× linear is ambitious.** At that ratio the model is inventing >90% of the output pixels. If quality at 10× is poor, the 5× (200→1000) path is the fallback production pipeline — still crisp on a normal monitor. Planned for explicitly in the dual demo targets.
- **VAE detail loss.** SD 1.5's VAE is lossy on fine high-frequency detail. Phase 2.5 tests alternative VAEs empirically and picks the best one to carry forward into Phase 4 training.
- **LoRA on `x4-upscaler` specifically is unusual.** Most public LoRAs target SD 1.5/SDXL text-to-image. The upscaler's extra LR input channel needs `peft` to hook cleanly — verify early in Phase 4b. Fallback: LoRA on the SD 1.5 refinement pass (stage B).
- **Data licensing.** Training data is public-domain or permissive-research-licensed (DIV2K, LSDIR). Published LoRA weights should cite datasets. Personal photos stay in test-set only so the test visuals are always copyright-clean. Evaluation math never moves image pixels outside your machine.
- **Cold-start UX.** First request after idle is 20–60 sec. Real effort on the "warming up" state — animated progress bar with phases, honest expectations.
- **Cloud training cost runaway.** DO H100 is $3.39/hr whether you're using it or just forgetting about it. Alert at $40, always `terminate` (not `stop`) droplets when done. Checkpoints and datasets live in DO Spaces, not on the instance. Running 4c and 4d back-to-back on a single droplet avoids one full hour of spin-up billing.
- **suppixel.ai capacity / cost.** Running 15 images × 3 ratios × 2 models = 90 renders through the workbench UI. If their pricing makes this prohibitive, fall back to 5 representative test images rather than skipping the comparison entirely. Note in the writeup either way.
- **SDXL deployment VRAM.** If SDXL wins Phase 4e, Modal GPU upgrades from A10G (24 GB) to L4/L40S. Cost per request roughly doubles but still within Modal free credit at expected traffic.
- **Model caching on serverless.** Every cold start loads ~4–8 GB of weights. Pre-populate the Modal Volume on first deploy; cache locally in the container on reuse. Verify early in Phase 6.
- **Captioning variance.** BLIP-2 captions are noisier than LLaVA's. If caption quality is visibly poor, either upgrade to a larger local VLM or pre-filter/rewrite a subset of captions by hand.

## 6. Reading list

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020) — read before Phase 3.5.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022) — SD foundation.
- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models* (2023) — ControlNet.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) — applies identically to U-Nets.
- **Yu et al., *Scaling Up to Excellence* (CVPR 2024) — SUPIR, included in this folder. Re-read §3.2 (restoration-guided sampling) and §4 (LLaVA prompting) before Phase 4.** This is the reference point the project is framed against.
- Li et al., *BLIP-2* (2023) — the captioning model.
- `diffusers` docs: [upscaling](https://huggingface.co/docs/diffusers/using-diffusers/img2img#super-resolution), [LoRA training](https://huggingface.co/docs/diffusers/training/lora).
- Modal docs: [serverless GPU functions](https://modal.com/docs/guide/gpu), [Volumes](https://modal.com/docs/guide/volumes).

## 7. Open questions to revisit at kickoff

- Which 10–15 personal test images? (Freeze the set in Phase 1; it becomes the benchmark throughout.)
- Is there a print-resolution follow-on you want to keep warm in the background — or strictly parked for after this project ships?
- Domain of the `SD_image_upscaler` demo if deployed to a subdomain?

## 8. Final deliverable: the SUPIR-comparison writeup

The README is not just install instructions. It's the project's narrative artifact. Proposed outline:

1. **What is SUPIR (and HYPIR).** One paragraph on the 2024 research SOTA (SUPIR: SDXL + 600M adaptor + LLaVA prompting + restoration-guided sampling, trained on LSDIR with research compute). One paragraph on HYPIR, SUPIR's commercially productized successor, now offered via suppixel.ai.
2. **What this project is.** A 50-hour consumer-GPU exploration using the same conceptual recipe: diffusion + LR conditioning + captions + LoRA. Reuses pretrained components; trains one LoRA. Compared honestly against both the research SOTA and its commercial descendant.
3. **Pipeline architecture** with the diagram from §2.
4. **Key architectural choices** — where we diverge from SUPIR and why.
5. **Leaderboard** at 4× / 5× / 10×: bicubic / Real-ESRGAN / our SD 1.5+LoRA / our SDXL+LoRA / SUPIR / HYPIR. LPIPS + DISTS + human rank.
6. **Gallery:** same test images, same crops, all methods side-by-side.
7. **Prompt-ladder results** — how much does prompting matter?
8. **VAE swap results** — where does detail loss actually happen?
9. **Cost and compute comparison.** Research-cluster-plus-LLaVA + 86k training images (SUPIR) vs. a commercial pipeline (HYPIR, inferred architecture) vs. your one GPU weekend + $40 cloud + 2000 training images.
10. **Hour log.** Honest breakdown of where the 50 hours went.
11. **What I'd do differently** / **what surprised me.**
12. **Reproducing this project.** Install/run in two modes — local `docker compose up` or Modal-backed deployment.

Proposed README subtitle for the repo: *"A diffusion image upscaler in 50 hours: how far can one person get against SUPIR and HYPIR on consumer hardware?"*

## 9. Brad's responsibilities

Claude Code owns the repo, the code, and most of the execution. Brad owns a small but critical set of tasks that Claude Code cannot do. Making these explicit here so nothing falls through the cracks.

**Pre-kickoff**
- Create/confirm Hugging Face account; generate a read token (write token only needed for Phase 4c/4d LoRA uploads).
- Create/confirm a Modal account (can wait until Phase 6 but painless to do early).
- Configure DO account spend alerts and any region prerequisites for GPU droplets.

**Phase 1**
- Curate a **60-image frozen test set**, each image cropped to a 1000×1000 square:
  - **30 traditional:** 10 `landscape`, 10 `animals`, 10 `cityscape`.
  - **30 hard:** cross-section of the six challenge categories (`text`, `fine_architecture`, `hf_texture`, `reflection`, `noise`, `night`).
- Copy into `data/test_images/`.
- Create `data/test_images/metadata.json` tagging each image with its category (and, for hard images, one or more challenge tags).
- **This is the frozen benchmark for the entire project** — the set is locked after Phase 1 acceptance. Spend real time on selection.
- Review Phase 1 output; approve before Phase 2 begins.

**Phase 2**
- Review 4× / 5× / 10× grids. Confirm the 5× path meets the "crisp on a monitor" bar.
- Approve.

**Phase 2.5**
- Eyeball VAE comparison on a fine-texture crop. Confirm or override Claude Code's pick.

**Phase 3**
- Produce a subjective human-rank of the methods on 3–5 images for the metric-vs-human correlation.
- Approve the preliminary leaderboard (baselines + our pre-LoRA pipeline across all 60 images).

**Phase 4a**
- Spot-check 20 random BLIP-2 captions for quality; flag if >30% look wrong.
- Provision DO Spaces bucket and credentials for the training-dataset mirror.

**Phase 4b**
- Review rehearsal loss curve + intermediate samples. Green-light the cloud runs.

**Phase 4c + 4d**
- **Provision the DO H100 droplet** (you, not Claude Code, pull the trigger on paid compute). Monitor for runaway costs. **Terminate** the droplet when done — not `stop`.
- Consider running 4c and 4d back-to-back on the same droplet instance.

**Phase 4e**
- Look at before/after grids. Make the **SD 1.5 vs SDXL deployment call** — this is an empirical judgment, not an algorithmic one.

**Phase 4.5**
- Rank the prompt levels on two images yourself. Confirm or challenge the empirical finding.

**Phase 4.6**
- Review the Phase 4e leaderboard; select the 12-image SUPIR/HYPIR comparison subset using the clear-win / clear-loss / surprise / coverage rubric. Record selection + rationale in `outputs/supir/subset.json`.
- **Run SUPIR + HYPIR through suppixel.ai manually** on the 12 images × 3 ratios × 2 models = 72 renders. Save outputs with consistent naming. Record credits / $ spent.
- Review the head-to-head analysis; confirm the "why we lost" hypotheses Claude Code drafts per clear-loss image.
- Approve.

**Phase 5**
- Play with the local Gradio UI. File UX issues *before* Phase 6.

**Phase 6**
- Provide droplet IP + SSH key path via `.env`.
- Optionally: point a subdomain at the droplet for the live URL.
- Monitor real cost for one week; record numbers for the writeup.

**Phase 7**
- Write the **"what I'd do differently"** and **"what surprised me"** sections yourself — those are your learnings, not Claude Code's.
- Do the final hour-log tally.

**Every phase**
- Review, push back, approve. That's the collaboration contract from §0.
