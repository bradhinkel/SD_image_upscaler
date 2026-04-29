# CLAUDE.md — SD_image_upscaler

This file is the operator manual for **Claude Code** working on this project. Read it in full before making any changes.

The authoritative project plan is `docs/project_plan.md` — it has the narrative, design decisions, and the full rationale. This file is the compact, action-oriented companion.

---

## Working mode — read this first

**Work on one phase at a time.** Do not skip ahead. Do not start Phase 2 while Phase 1 is still open.

For each phase:

1. Read the phase's section in `docs/project_plan.md` in full.
2. Read the phase's **acceptance criteria** (below in this file, or mirrored in the plan).
3. Do the work. Make regular commits with clear messages.
4. When you believe the acceptance criteria are met, **stop**. Do not continue to the next phase.
5. Produce a short phase-completion summary for Brad: what was built, what the verification output looks like, any deviations from the plan and why, what to look at first when reviewing.
6. **Wait for Brad to review and explicitly approve the phase before starting the next one.** He may push back, ask for changes, or reshape the plan. That's expected.

If you hit something in the plan that's wrong, ambiguous, or would need a meaningful deviation, **stop and ask** rather than guessing. The plan was shaped iteratively; it's meant to be revisable.

Do not modify `docs/project_plan.md` without being asked.

---

## Project overview

A diffusion-based image upscaler built as a 50-hour capability study against SUPIR (CVPR 2024). Primary target: 200×200 → 1000×1000 (5× linear). Secondary: 250×250 → 1000×1000 (4×). Stretch: 100×100 → 1000×1000 (10×) — **aspirational**, run and report, no quality bar.

Subject domain: landscape, architecture, cityscape, and animal photography.

License: MIT.

Key decisions:
- Base model: SD 1.5 trained first; SDXL trained only if Phase 4c passes the gate (see Phase 4c criteria). Final deployment model decided empirically in Phase 4e.
- Backbone: `stable-diffusion-x4-upscaler` (stage A) + `ControlNet Tile` (stage B).
- Training: one LoRA per base model (SD 1.5 mandatory, SDXL conditional), captioned training data (BLIP-2).
- Training compute: **RunPod RTX 5090 32 GB at ~$0.69/hr (Community Cloud), provisioned by Brad for each run**. Same Blackwell architecture as the local 5070 laptop, so the cu128 torch wheels in `pyproject.toml` run on the pod unchanged. Dataset transfer via HuggingFace Hub (private dataset repo).
- Deployment: DO droplet frontend (4 GB, no GPU) + Modal serverless GPU backend. Default Modal GPU = A10G (revisit if SDXL wins the 4e decision).
- Reference benchmarks: **SUPIR + HYPIR**, both run manually by Brad through suppixel.ai. Claude Code does not attempt to run these itself — it only consumes and scores the downloaded outputs.

See `docs/project_plan.md` §1 for the decision table.

---

## Repository conventions

- **Package-first.** Real logic lives in `src/upscaler/`. Notebooks in `notebooks/` are thin consumers — they `import upscaler.*`, orchestrate, and display. Phase 3.5 (`035_tiny_diffusion.ipynb`) is the one deliberate exception: DDPM logic stays inline with markdown explanation.
- **Scripts** in `scripts/` are CLI wrappers over `upscaler/` for reproducibility.
- **Tests** in `tests/` are deterministic and GPU-free. They should pass in CI on a CPU runner.
- **Configs** for training runs live in `configs/*.yaml`.
- **Outputs** go to `outputs/` (gitignored). Never commit generated images, LoRA checkpoints, or large result files.
- **Data** goes to `data/` (gitignored). Never commit dataset files or personal photos.
- The **frozen test set** is chosen in Phase 1 and never changed. Do not add, remove, or reshoot test images after Phase 1 is accepted.

---

## Commands — run these often

All commands assume WSL2 Ubuntu with the project's `uv` environment activated.

### Environment
```bash
# One-time setup (Phase 0)
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"   # dev deps: pytest, ruff, pre-commit

# Verify GPU is visible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Quality gates (run before every commit, and as part of phase completion)
```bash
uv run ruff format .                    # autoformat
uv run ruff check . --fix               # lint with autofixes
uv run pytest                           # full test suite
uv run pytest -x -vv                    # fail fast with verbose output when debugging
```

### Running the pipeline locally
```bash
# Single image via script
python -m scripts.run_inference data/test_images/img01_200.jpg --scale 5 --out outputs/

# Training rehearsal (Phase 4b)
python -m upscaler.lora_train --config configs/rehearsal.yaml

# Dataset build (Phase 4a)
python -m scripts.build_training_pairs --hr-dir data/hr --out data/pairs --crop 512
python -m scripts.caption_dataset --pairs data/pairs --out data/captions.jsonl

# Benchmark run (Phase 3)
python -m scripts.benchmark_pipeline --methods all --out outputs/eval/
```

### Frontend
```bash
# Local Gradio (Phase 5)
uv run python frontend/main.py

# Full stack via docker (Phase 5 deliverable)
docker compose up
```

### Modal deployment (Phase 6)
```bash
modal deploy src/upscaler/modal_app.py
modal app list
modal app logs sd-image-upscaler
```

---

## Acceptance criteria per phase

Each phase is "done" only when **every** criterion below passes. Produce verification output (command + result) for each.

### Phase 0 — Environment and package scaffold
- [ ] Repo pushed to GitHub as `SD_image_upscaler` with MIT `LICENSE` at repo root.
- [ ] `uv pip install -e .` completes without error.
- [ ] `uv pip install -e ".[dev]"` completes without error.
- [ ] `nvidia-smi` inside WSL shows the GPU.
- [ ] `python -c "import torch; assert torch.cuda.is_available()"` exits 0.
- [ ] `uv run ruff check .` exits 0.
- [ ] `uv run pytest` exits 0 (may be "no tests collected" — that's fine).
- [ ] GitHub Actions CI workflow exists and has succeeded on at least one push.
- [ ] `CLAUDE.md` and `.env.example` both committed at repo root.
- [ ] `notebooks/00_smoke.ipynb` (or equivalent) runs end-to-end: imports `upscaler`, generates one SD 1.5 text-to-image at 512×512.
- [ ] Pre-commit hook installed and `pre-commit run --all-files` passes.

### Phase 1 — Baseline upscaler
- [ ] `src/upscaler/pipeline.py` defines `UpscalerPipeline` with `.load()`, `.close()`, `.upscale_x4(image, prompt, noise_level, steps) -> Image`.
- [ ] `src/upscaler/baselines.py` exposes `bicubic`, `lanczos`, `realesrgan`.
- [ ] `src/upscaler/testset.py` (or equivalent) exposes a helper to load test-set metadata and yield images by slice.
- [ ] `tests/test_pipeline_smoke.py` passes on CPU using a mocked diffusers pipeline (≥3 test cases).
- [ ] `tests/test_testset.py` covers metadata loading and slice queries.
- [ ] `data/test_images/` contains **60 images** (1000×1000 squares): 30 traditional (10 `landscape`, 10 `animals`, 10 `cityscape`) + 30 hard (~5–8 per challenge tag: `text`, `fine_architecture`, `hf_texture`, `reflection`, `noise`, `night`; multi-tag images allowed). Source images in `Images/Test_Images/` are 1024×1024 and must be **center-cropped** to 1000×1000 during migration (no resize).
- [ ] Each image has `_100.jpg`, `_200.jpg`, `_250.jpg` LR variants (10× / 5× / 4× sources respectively), generated via **bicubic downsampling** from the 1000×1000 HR.
- [ ] `data/test_images/metadata.json` exists and tags every image with `category` and — for hard images — one or more `challenges`. The existing `metadata.json` carries over verbatim; do not regenerate.
- [ ] Test set is **frozen** — no additions or removals after this phase.
- [ ] `notebooks/01_baseline.ipynb` runs end-to-end (Kernel → Restart & Run All) with no errors.
- [ ] `outputs/phase1/` contains **per-slice contact sheets** (one per traditional subcategory, one per challenge tag). A single all-60-images contact sheet would be unusable.

### Phase 2 — Two-stage pipeline at 4× / 5× / 10×
- [ ] `UpscalerPipeline.upscale_two_stage(image, target_size, denoise, steps, cn_weight, prompt) -> Image` implemented.
- [ ] `src/upscaler/tiling.py` exposes pure-function tile/merge logic.
- [ ] `tests/test_tiling.py` has ≥5 tests and all pass: tile-grid correctness, blend weights sum to 1 in overlap, round-trip identity returns input byte-for-byte, edge-tile handling, non-square input handling.
- [ ] `notebooks/02_two_stage.ipynb` produces, for each test image: a grid of 4× / 5× / 10× outputs × several denoise strengths.
- [ ] Output grids saved to `outputs/phase2/`.
- [ ] At least one grid shows the fidelity cliff between 5× and 10× clearly enough to annotate.

### Phase 2.5 — VAE swap
- [ ] `notebooks/025_vae_swap.ipynb` runs end-to-end, comparing SD 1.5 default / SDXL backport / sd-vae-ft-mse / taesd on the full test set.
- [ ] `outputs/phase2_5/vae_comparison.csv` exists with per-image LPIPS for each VAE.
- [ ] Notebook includes a markdown cell with the chosen VAE to carry forward into the **SD 1.5 track** (Phase 4c) and one-sentence justification. The SDXL track (Phase 4d) defaults to SDXL's native VAE and does its own small A/B against this winner; this notebook does not decide for SDXL.

### Phase 3 — Evaluation harness + preliminary leaderboard
Phase 3 produces the first leaderboard covering baselines + our pre-LoRA pipeline across the full 60-image set. SUPIR/HYPIR comparison is **deferred to Phase 4.6** so the subset selection can be informed by Phase 4e's LoRA results.

- [ ] `src/upscaler/eval_metrics.py` exposes `lpips`, `dists`, `psnr`, `ssim`, `evaluate_method`.
- [ ] `tests/test_eval_metrics.py` passes and covers both "identical input" and "random input" edge cases per metric.
- [ ] `scripts/benchmark_pipeline.py` is runnable: `python -m scripts.benchmark_pipeline --methods all --out outputs/eval/` produces a CSV.
- [ ] `outputs/eval/leaderboard_phase3.csv` has rows for bicubic, Lanczos, Real-ESRGAN, our two-stage (no LoRA) across the **full 60-image set** × each ratio — with LPIPS + DISTS + PSNR + SSIM.
- [ ] `notebooks/03_eval.ipynb` renders leaderboard bar charts sliced by (a) overall, (b) traditional vs hard, (c) per challenge tag. Plus gallery and metric-vs-human correlation on 3–5 images.

### Phase 3.5 — Tiny diffusion from scratch
- [ ] `notebooks/035_tiny_diffusion.ipynb` runs end-to-end on local GPU.
- [ ] Training completes 5k steps in <4 hours; loss curve decreases monotonically after early warmup.
- [ ] At least 8 markdown cells explaining concepts (forward process, noise schedule, loss, reverse sampling, etc.).
- [ ] Final model produces recognizable-but-bad 64×64→256×256 outputs for 3 sample inputs.

### Phase 4 — Captioning + LoRA training
**4a.** Dataset + captioning

Source images (Brad delivers to `data/raw/`):
| Folder | Count | Source | Filter | Min short side |
|---|---|---|---|---|
| `data/raw/div2k/` | 800 | DIV2K train HR | (all) | native |
| `data/raw/unsplash_landscape/` | 600 | Unsplash Lite | landscape / nature | ≥2048 px |
| `data/raw/unsplash_cityscape/` | 600 | Unsplash Lite | cityscape / architecture | ≥2048 px |
| `data/raw/unsplash_animals/` | 600 | Unsplash Lite (preferred) or iNat-2021 | animals | ≥2048 px |

Pair-build pipeline (Claude Code):
- 3 random 512×512 crops per source → ~7800 HR tiles.
- LR_128 generated from HR_512 via the realistic degradation module (NOT plain bicubic).
- Perceptual-hash dedup across training crops; near-dupes evicted.
- Perceptual-hash leakage check against the 60 frozen test images; any match is evicted from training.

- [ ] `src/upscaler/degradations.py`, `captioning.py`, `dataset.py` implemented.
- [ ] `tests/test_degradations.py` passes with ≥4 test cases.
- [ ] `data/raw/` populated by Brad per the table above (~2600 source images).
- [ ] `data/pairs/` contains ≥2000 (LR_128, HR_512) pairs (target: ~7800).
- [ ] `data/captions.jsonl` has one caption per HR tile.
- [ ] 20 random captions sampled and visually checked — no obvious garbage (>30% wrong → filter or upgrade VLM).
- [ ] Dedup + test-set leakage check run. `outputs/phase4/leakage_report.md` written: lists dedup count, any near-dupes against the frozen test set with perceptual-hash distance and file pairs, and a summary (rendered for Brad review before training).
- [ ] Dataset (pairs + captions) pushed to a **private** HuggingFace Hub dataset repo (`bradhinkel/sd-image-upscaler-pairs`) so the RunPod pod can pull it via `datasets.load_dataset` at training time.

**4b.** Local rehearsal
- [ ] `src/upscaler/lora_train.py` runnable: `python -m upscaler.lora_train --config configs/rehearsal.yaml`.
- [ ] Rehearsal completes in <3 hours locally; loss decreases; intermediate samples rendered in `outputs/runs/rehearsal/`.

**4c.** SD 1.5 run on RunPod RTX 5090 pod — **CLOSED with negative result**

The originally planned x4-upscaler LoRA was attempted three times (~$2.55 cloud) and produced catastrophically destructive deltas at any usable scale (LPIPS 0.79–0.92 vs base 0.33). Diagnostic confirmed the LoRA mechanism was loading correctly but the trained weights pushed the denoising trajectory off the natural-image manifold. Re-reading the SUPIR paper validated this as a structural issue: SUPIR explicitly avoids x4-upscaler as a base, uses zero-init additive adapters (not LoRA), and targets non-attention layers.

The pivot was a stage-B SD 1.5 LoRA targeting cross-attention (`bradhinkel/sd-image-upscaler-sd15-lora`, public, with an honest model card flagging the niche-only utility). Training was clean (no destabilisation), but the LoRA does **not** improve the two-stage pipeline on average:

| Pipeline | Mean LPIPS at 5× |
|---|---|
| Real-ESRGAN | 0.299 |
| Two-stage (no LoRA) | 0.433 |
| Two-stage + stage-B LoRA | 0.443 |

Per-category, the LoRA helps night scenes (62.5% win rate, mean −0.006 LPIPS) and is roughly neutral on reflections; it mildly regresses everywhere else.

Total Phase 4c cloud spend: **$3.40** across all 4 training attempts (well under the $10 cap).
Total Phase 4c wall-clock: ~5.5 hrs (well under 6 hrs).

Acceptance for closure:
- [x] At least one LoRA training run completed successfully (the stage-B run).
- [x] LoRA checkpoint saved to `outputs/loras/sd15_stage_b/` and pushed to HF Hub at `bradhinkel/sd-image-upscaler-sd15-lora`.
- [x] Training logs saved to `outputs/runs/sd15_stage_b/`.
- [x] **RunPod pod terminated** by Brad after eval completed.
- [x] Post-training quick eval on full test set at 5× completed (`scripts/eval_lora_stage_b.py` → `outputs/eval/lora_stage_b_gate.csv`).
- [x] Negative-result writeup material captured for Phase 7.

**Publishing policy (applied):**
- LoRA + honest model card pushed publicly. Card includes the unflattering numbers, the night-scene niche, and a "read this before using" note pointing to the project repo.
- Unsplash Lite TERMS.md re-checked at training time; ML training is permitted under the Lite Dataset terms.

**SDXL gate decision (was: proceed to 4d if ≥50% LPIPS win over Real-ESRGAN + <$10 + <6 hrs):**

Gate **FAILED** on criterion 1: stage-B + LoRA beat Real-ESRGAN on **3.3% (2/60)** of test images. Two readings of the failure:
1. Cross-attention LoRA is the wrong adapter type for restoration (SUPIR's lesson).
2. Two-stage diffusion isn't the right tool when LR is clean bicubic — Real-ESRGAN's SR-specific GAN already extracts what's there.

**4d.** SDXL run on RunPod — **SKIPPED**

The SDXL gate failure means a larger SDXL LoRA run would compound the same architectural mismatch on a more expensive base. The marginal cloud spend (~$10) and wall-clock (~10 hrs) is better invested in Phase 4.6's head-to-head and Phase 7's writeup. The Phase 7 follow-on roadmap describes the SDXL-base path as Tier 2 of a possible next project.

**4e.** Evaluation — **subsumed into Phase 4c closure + Phase 4.6.**

Originally Phase 4e was to merge SD 1.5+LoRA and SDXL+LoRA leaderboard rows into `outputs/eval/leaderboard_phase4.csv` and pick a winner. With 4d skipped, the relevant comparisons are already in `outputs/eval/lora_stage_b_gate.csv` (no-LoRA two-stage vs LoRA two-stage vs Real-ESRGAN, full 60-image set). Phase 4.6 will merge SUPIR + HYPIR rows into a single final leaderboard.

### Phase 4.5 — Prompting ladder
- [ ] `notebooks/04_prompt_ladder.ipynb` runs all six prompt levels × full test set.
- [ ] `outputs/phase4_5/prompt_comparison.csv` has per-image, per-prompt-level LPIPS values.
- [ ] Hero "same LR, five wildly different prompts" demo rendered for ≥2 images.
- [ ] Notebook ends with a markdown summary: "prompting matters a lot / a little / varies by domain because…"

### Phase 4.6 — SUPIR / HYPIR benchmark (4× only)

Data-driven comparison: subset selection informed by Phase 4c's per-image LPIPS results. Brad runs SUPIR + HYPIR manually through suppixel.ai.

**Scope note:** suppixel.ai's SUPIR and HYPIR support upscaling at 1×/2×/3×/4× only — no 5× or 10× capability. This matches the broader landscape (Real-ESRGAN, x4-upscaler are also 4× native). Phase 4.6 is therefore the **4× head-to-head**; our 5× and 10× results are framed in Phase 7 as exploration beyond what current public diffusion-SR tools support, with no available SOTA reference at those ratios.

- [x] `outputs/supir/subset.json` documents 12 selected images with a per-image rationale (clear win / clear loss / surprise / coverage).
- [ ] **Brad has run** SUPIR + HYPIR through suppixel.ai: 12 images × 1 ratio (4×) × 2 models = **24 renders**. LR inputs are `data/test_images/{stem}_250.jpg` (the 4× variant: 250 → 1000). Outputs land in `outputs/supir/{stem}_4x.{jpg|png}` and `outputs/hypir/{stem}_4x.{jpg|png}`.
- [ ] Credits / $ spent on suppixel.ai recorded for the cost log.
- [ ] `outputs/eval/leaderboard_phase4_6.csv` merges the 24 SUPIR + HYPIR rows with the existing 4× rows from `leaderboard_phase3.csv` for those 12 images.
- [ ] `notebooks/06_supir_head_to_head.ipynb` renders 12 side-by-side grids at 4× (HR / bicubic / Lanczos / Real-ESRGAN / our two-stage / SUPIR / HYPIR), one per image, with per-image commentary noting the SOTA gap.
- [ ] Notebook ends with a markdown summary: where we're competitive at 4×, where the gap is largest, and the takeaway that going beyond 4× is research territory rather than a comparison gap.

### Phase 5 — Local Gradio app
- [ ] `frontend/main.py` runs: `uv run python frontend/main.py` → Gradio at `localhost:7860`.
- [ ] UI exposes: image upload, target-ratio dropdown (4×/5×/10×), denoise slider, steps slider, prompt textbox, LoRA toggle, seed input.
- [ ] Output shows LR input + stage A intermediate + stage B final.
- [ ] `frontend/Dockerfile` builds clean.
- [ ] `docker-compose.yml` at repo root.
- [ ] `docker compose up` brings up the full app on a local NVIDIA+Docker host; tested by running an actual upscale through it end-to-end.

### Phase 6 — Deployment
- [ ] `src/upscaler/modal_app.py` deploys: `modal deploy src/upscaler/modal_app.py` succeeds. Default GPU: A10G (SD 1.5 demo). If the 4e decision picked SDXL, re-evaluate — A100 may be warranted.
- [ ] Cold start measured <90 sec with Modal Volume cache warm; warm inference <20 sec.
- [ ] `frontend/main.py` updated to call Modal (mode configurable via env: `UPSCALER_MODE=local|modal`).
- [ ] Frontend deployed to the 4 GB DO droplet via `docker compose`.
- [ ] Live URL reachable over HTTPS with valid cert.
- [ ] Rate limiter enforces 3 req/IP/hour; daily cap enforced; both tested with intentional overruns.
- [ ] README has two reproduction paths: local `docker compose up` and Modal-backed.
- [ ] Actual monthly cost measured over 1 week; reported in the README.

### Phase 7 — Writeup
- [ ] README rewritten per `docs/project_plan.md` §8 outline.
- [ ] Final leaderboard CSV at `outputs/eval/final_leaderboard.csv`.
- [ ] Hero images embedded in the README.
- [ ] Hour log + cost log sections filled out with real numbers.
- [ ] At least 3 entries in "what I'd do differently" / "what surprised me."

---

## Environment variables

All secrets live in `.env` at repo root (never committed). Copy `.env.example` → `.env` and fill in.

See `.env.example` for the full list. Short version of what you'll need by phase:

| Phase | Vars needed |
|---|---|
| 0–3 | `HF_TOKEN` (for model weight downloads) |
| 3 | none (SUPIR/HYPIR via suppixel.ai is manual; Brad downloads outputs) |
| 4a | `HF_TOKEN` (write scope — used to push the dataset to a private HF Hub repo) |
| 4c/4d | `RUNPOD_API_KEY` (pod provisioning), `HF_TOKEN` (dataset pull + LoRA push from inside the pod) |
| 6 | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` + `DO_DROPLET_IP` / `DO_SSH_KEY_PATH` + optional `DEMO_DOMAIN` |

---

## Hard rules — never do these

- **Never commit** secrets, `.env`, HF tokens, or model weights.
- **Never commit** generated images, LoRA checkpoint files, dataset pairs, or anything in `outputs/` or `data/`. These are gitignored; keep them that way.
- **Never modify the frozen test set** (`data/test_images/`) after Phase 1 acceptance. New images would invalidate all downstream comparisons.
- **Never skip `pytest` + `ruff check`** before declaring a phase complete.
- **Never amend or force-push** a commit that's already been pushed, unless Brad explicitly asks.
- **Never train on** or ingest personal photos into training data — they stay test-set only.
- **Never spin up cloud GPU** without a spend alert configured and a clear termination plan. Use `terminate`, not `stop`, after runs.
- **Never invent an acceptance-criteria result.** If a criterion can't be verified, say so and ask.

---

## Gotchas and sharp edges

- **CUDA in WSL2:** the driver is installed on the Windows *host*, not inside WSL. Only the CUDA *toolkit* goes inside WSL. If `nvidia-smi` fails inside WSL, the host driver is the wrong version or WSL's GPU passthrough isn't enabled.
- **`xformers` version:** must match the installed `torch` version *exactly*. If `pip install xformers` upgrades torch unexpectedly, pin both in `pyproject.toml`.
- **RAM spillover behavior:** when VRAM is exhausted, CUDA silently swaps to system RAM and step time jumps 5–30×. For training, this is acceptable overnight; for interactive work, reduce batch size instead.
- **Modal Volume cold-populate:** the *very first* deploy writes ~4–8 GB to the volume, which takes 3–5 minutes and counts against your cold-start time. Subsequent cold starts are ~30 sec. Plan the first deploy accordingly.
- **Diffusers cache:** `HF_HOME` or `HUGGINGFACE_HUB_CACHE` controls where weights are cached. Point it at a large-volume mount, not `/root` or `/home`, so you don't fill up the OS disk.
- **LoRA on `x4-upscaler`:** the U-Net has an extra LR input channel vs. text-to-image SD. Verify `peft` attaches cleanly in Phase 4b (rehearsal) before spending cloud $. Fallback: LoRA on the stage-B SD 1.5 pass instead.
- **SDXL on 8–10 GB VRAM:** inference only works with fp16 + offloading, and even then it's tight. Don't attempt SDXL *training* locally — that's cloud only. Inference locally is fine for development.
- **BLIP-2 captions can be noisy.** Spot-check 20 captions after Phase 4a. If >30% look wrong, either filter the dataset or upgrade to a larger VLM.

---

## What Brad owns (not Claude Code)

Full list is in `docs/project_plan.md` §9. Summary:

- **Phase 1:** curating the 60-image frozen test set (30 traditional + 30 hard) and authoring `metadata.json`.
- **Phase 3:** producing a subjective human-rank of 3–5 images for metric-vs-human correlation.
- **Phase 4a:** spot-checking BLIP-large captions; creating the HF Hub access token (write scope).
- **Phase 4c / 4d:** provisioning the RunPod RTX 5090 pod, monitoring cost, terminating when done.
- **Phase 4e:** the SD 1.5 vs SDXL deployment judgment call.
- **Phase 4.6:** selecting the 12-image SUPIR/HYPIR comparison subset from Phase 4e leaderboard data; running SUPIR + HYPIR manually through suppixel.ai and depositing outputs.
- **Phase 5:** hands-on UX review of the Gradio app.
- **Phase 6:** droplet IP/SSH provisioning, optional domain setup, one-week cost measurement.
- **Phase 7:** the "what I'd do differently" / "what surprised me" writeup sections.
- **Every phase:** reviewing, pushing back, and explicitly approving.

If a task looks like it needs Brad's judgment or a credential only he has, that's a stop-and-ask situation. Don't work around it.

## When in doubt

1. Reread the relevant phase in `docs/project_plan.md`.
2. Check the acceptance criteria for that phase above.
3. If it's still ambiguous or the plan seems wrong, **stop and ask Brad** rather than making an irreversible choice.
