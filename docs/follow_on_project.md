# Follow-on project: closing the SUPIR gap incrementally

Two parallel paths emerged from the 50-hour study, distinguished by a clean feasibility line:

- **Tier 1 — feasibility-confirmed.** A practical desktop tiling upscaler built on Real-ESRGAN, with optional diffusion-refinement layered on top. Reuses everything this study already shipped (`tiling.py`, `pipeline.UpscalerPipeline`, the stage-B LoRA, the Real-ESRGAN baseline). Zero cloud spend, ~25 engineering hours, and produces a tool people would actually use.
- **Tiers 2–4 — research-flavored, SUPIR exploration track.** Progressively-larger investments aimed at closing the SOTA gap. Each tier targets a specific architectural lesson the failed LoRA training attempts surfaced. Tier 4 is a real SUPIR-class rebuild on public tools.

The capability study established what's feasible at consumer-GPU scale (Tier 1) and what isn't without research-scale data and engineering (Tiers 2–4). Both are worth pursuing; pick the one that matches the available time and budget.

Each tier's value is tested by a clear empirical question that the previous tier didn't answer.

---

## Tier 1 — Real-ESRGAN desktop tiling upscaler with optional diffusion refinement, ~25 hrs, $0 cloud

**Core question: what's the best upscaler one user can ship on their own machine?**

This is the practical follow-on. The capability study showed Real-ESRGAN at 4× sits within ~6% of SUPIR's quality on the standard LPIPS benchmark — at a fraction of the inference cost and with no API dependency. **A well-engineered tiled desktop tool around Real-ESRGAN (with optional diffusion refinement for users who want it) is the realistic ship-able product** that emerges from this study.

### What it is

A standalone Gradio (or Tauri/Electron) app:

- **Primary path: Real-ESRGAN x4 tiled.** `realesrgan` package's `RealESRGANer` class handles tiling natively (`tile=512, tile_pad=10`, fp16). ~5 GB VRAM at tile=512; runs on basically any GPU including 8 GB consumer cards. Output: any input → 4× the size, deterministic.
- **Optional toggle: 8× / 16× via two-pass x4.** Cleaner output than the single-pass x8 model in nearly every test; one-button selection.
- **Optional toggle: "Add diffusion refinement."** A second pass through SD 1.5 img2img + ControlNet Tile at `denoise=0.2`, with our published stage-B LoRA optionally applied for night-scene preference. Doubles processing time but adds diffusion-style perceptual detail.
- **Per-image LPIPS scoring** if a reference HR is provided (so users can benchmark their own workflows the same way this study benchmarked methods).
- **Streaming tile management** for arbitrary-size inputs — 4K → 16K should not require 16K² of VRAM at any moment.

### What it reuses from this study

| Capability | Source |
|---|---|
| Tiled inference with byte-for-byte-stable blend | `src/upscaler/tiling.py` (Phase 2) |
| Real-ESRGAN integration via `spandrel` | `src/upscaler/baselines.py` |
| Two-stage diffusion pipeline + LoRA injection | `src/upscaler/pipeline.py` (Phases 2 + 4c) |
| Stage-B SD 1.5 LoRA artifact | `bradhinkel/sd-image-upscaler-sd15-lora` on HF Hub |
| LPIPS / DISTS metric infrastructure | `src/upscaler/eval_metrics.py` (Phase 3) |

This isn't from-scratch work. It's gluing the existing `upscaler` package into a polished frontend with sensible defaults, memory-aware tiling, and the diffusion-refinement toggle that gives the published LoRA a real product-style use case beyond its current "research artifact" framing.

### Effort breakdown

| Task | Hours |
|---|---|
| Gradio frontend (image upload, parameter UI, progress display) | 8 |
| Streaming tile management for arbitrary-size inputs | 4 |
| 8×/16× two-pass logic + intermediate-storage handling | 3 |
| Diffusion-refinement integration (toggle + LoRA toggle) | 5 |
| Per-image LPIPS scoring UI + reference-image upload | 2 |
| Packaging (Dockerfile + optional desktop-app shell) | 3 |
| **Total** | **~25 hrs** |

Cloud cost: **$0**. The whole thing runs on the user's own GPU.

### Risks / caveats

- Memory management for very-large outputs (16K+) requires careful tile streaming. Real-ESRGAN's built-in tiling handles per-tile fine; the orchestrator layer needs to handle the tiling + output assembly without ever holding the full output in memory.
- The diffusion-refinement second pass is slow on 8 GB GPUs (~20–60 seconds per 1000² output for our two-stage). Users who turn it on need patience or a better GPU. Default state should be "off."
- Real-ESRGAN x8 model is in the repo but produces visibly worse results than two-pass x4. Mention in docs but default to two-pass.

**This is the project that should ship first.** It validates the capability study's main practical takeaway and produces a real artifact users can install.

---

## Tier 2 — ZeroSFT-style adapter, ~15 hrs, ~$10 cloud

**Core question: was the *adapter type* the issue in our LoRA training failure?**

What changes vs the current pipeline:
- Replace LoRA-on-cross-attention with a **zero-initialised additive adapter** on intermediate ResBlock features (after each ResBlock in stage B's SD 1.5 decoder). Same place SUPIR's ZeroSFT lives.
- Initial weights produce **exact zero delta** at training step 0 — base model behaves identically until training pushes the adapter weights, no destabilisation risk.
- Same 7,786-pair training data, same captions, same RunPod 5090 setup.

What stays:
- SD 1.5 stage-B base, ControlNet Tile structural conditioning, two-stage Phase 2 pipeline.
- BLIP-large captions (until Tier 3).
- Test set, eval scripts, leaderboard infrastructure.

**Implementation sketch (~10 hrs):**

```python
# src/upscaler/zero_sft.py — new module
class ZeroSFT(nn.Module):
    """Zero-init Spatial Feature Transform: scale + shift modulation
    on a residual feature map, conditioned on a control signal.
    Output is exactly 0 at init."""
    def __init__(self, ch, control_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        # Both projections init to zero -> output exactly 0 at step 0.
        self.scale_proj = nn.Conv2d(control_dim, ch, 1)
        self.shift_proj = nn.Conv2d(control_dim, ch, 1)
        nn.init.zeros_(self.scale_proj.weight); nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight); nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x, control):
        scale = self.scale_proj(control)
        shift = self.shift_proj(control)
        return self.norm(x) * (1 + scale) + shift
```

Hook one ZeroSFT after each of stage B's down/up ResBlocks (~12 attachment points). Train alongside the existing pipeline; loss target same as current (MSE on predicted noise).

**Key empirical question:** does the same training recipe on a zero-init adapter produce something that *actually improves* the pipeline? Specifically, does it move the 5× LPIPS from ~0.43 (current two-stage) toward ~0.30 (Real-ESRGAN territory)?

**Risks / caveats:**
- The "control" input to ZeroSFT is non-trivial — SUPIR feeds it from the restoration encoder. Cleanest first try: bicubic-upsample the LR to the latent's spatial size and pass directly.
- ZeroSFT adds parameters proportional to UNet feature widths. ~5–10M trainable params at the small end (vs LoRA's 3M); still tractable on the 5090.

**Estimated effort:** 10 hrs eng + 1 hr cloud training (~$0.85) + 2 hrs eval / writeup. **Total ~15 hrs / $10.**

---

## Tier 3 — SDXL base + LSDIR-scale data + LLaVA captions, ~50 hrs, ~$50 cloud

**Core question: was the *base model + recipe + scale* the issue?**

What changes vs Tier 2:
- **Switch base from SD 1.5 to SDXL.** SUPIR's base; longer-trained, larger UNet, stronger prior.
- **Pull LSDIR proper.** Training data jumps from 2,600 sources → 50,000–100,000 sources, ~5× to 10× our current scale.
- **Caption with LLaVA** (or any larger VLM than BLIP-large) — SUPIR's specific Phase 4 finding.
- **Train longer**: 30,000–50,000 steps vs current 8,000.

What stays:
- ZeroSFT-style adapter from Tier 2 (now validated as the right architecture choice).
- Two-stage pipeline; ControlNet Tile remains for structural conditioning.

**Implementation effort breakdown:**
- LSDIR download + curation: ~5 hrs (LSDIR is gated on HF; need access request + ~150 GB disk).
- LLaVA captioning of ~50k images: ~6 hrs runtime on the 5090, plus prompt-engineering iteration on what to ask LLaVA for.
- SDXL pipeline integration in our codebase (replacing SD 1.5 in stage B; ControlNet Tile has SDXL variants): ~6 hrs.
- Adapter retrofit for SDXL UNet (different ResBlock counts, channel widths): ~4 hrs.
- Cloud training: SDXL is ~2.5× larger than SD 1.5; expect 30k steps to take ~12–18 hrs at $0.69/hr → ~$10–15. With caption captioning + VAE handling, total cloud ~$20–30.
- Eval + writeup: ~6 hrs.

**Estimated effort:** ~35 hrs eng + ~15 hrs unattended cloud time. **Total ~50 hrs / $50 cloud + LSDIR access.**

**Key empirical question:** with a SUPIR-shaped recipe at 1/10th the data scale and 1/100th the compute, how much of SUPIR's ~3% advantage over Real-ESRGAN can we recover? Are we still at ~+15% LPIPS, or do we close to ~+5%?

**Risks:**
- SDXL training is meaningfully tighter on 32 GB VRAM. May need batch=2 + gradient accumulation, gradient checkpointing.
- LLaVA captions for 50k images take real time; budget 6+ hours on the 5090 with reasonable batching.
- LSDIR's gated access is a process step that adds project-time uncertainty.

---

## Tier 4 — Full SUPIR-style restoration encoder, ~200 hrs, ~$300 cloud

**Core question: can we approach SUPIR with public tools at proper training scale?**

What changes vs Tier 3:
- **Full SUPIR architecture port:** restoration encoder (separate from the SDXL UNet) trained jointly with the ZeroSFT adapter, mixed real+synthetic degradation curriculum, restoration-guided sampling at inference.
- **Multi-stage curriculum:** start with synthetic degradations (current pipeline), progressively introduce real wild-image degradations.
- **Larger training data:** 100k+ HR images, mixed source domains, careful filtering.

What's a real engineering project, not just config tuning:
- Implement the restoration encoder (SUPIR §3.1) — this is the bulk of the eng cost. Reference open-source SUPIR releases exist but are code-only; our wrapping into a pipeline + reproducible recipe is significant.
- Implement restoration-guided sampling (SUPIR §3.2) — the fidelity-quality tunable parameter. Modifies the diffusers pipeline's `__call__`.
- Curate / acquire wild-image training set — moves the project into "real research engineering" territory.

**Estimated effort:** ~150 hrs eng + ~50 hrs cloud (multiple long training runs at $0.69-$2.99/hr depending on GPU choice). **Total ~200 hrs / $300 cloud.** Multi-month side project, not a weekend.

**What this earns:**
- A from-scratch reproducible SUPIR-class model on public tools.
- Real per-paper benchmarks (LPIPS / DISTS / FID on standard SR benchmarks: DIV2K val, Set5/14, RealSR).
- Genuine SOTA contention if executed well.

**Where it'd likely fall short:**
- SUPIR's published quality came from research-cluster data + manual prompt curation that's hard to fully reproduce.
- 100k images < SUPIR's full training set; expect a quality gap.

---

## Cross-tier follow-up tasks

Independent of which tier is picked, three cheap improvements would help any of them:

1. **Implement Phase 4.5 (prompt ladder).** The original plan had this and it got cut. Six prompt levels × 60 images = 360 inferences ≈ 6 GPU-hours total. The Phase 4.6 caption A/B was a 12-image one-engine slice; a full ladder on our two-stage pipeline would give a generalisable answer to "how much does prompting matter for our recipe."

2. **Wild-image test set.** Our 60-image set is bicubic-LR-only — exactly the regime where Real-ESRGAN dominates. A 30-image *real-world LR* set (phone photos, JPEG-recompressed images, low-light shots) would shift the comparison toward where SUPIR was actually designed to operate. Diffusion-SR's value proposition lives there, not on bicubic.

3. **A second metric for human-perception correlation.** Phase 3's correlation experiment used LPIPS, DISTS, PSNR, SSIM. Adding **DISTS-trained or Toolkit-MUSIQ** would expand the Phase 3 methodological finding. If the project is going to be cited for the "PSNR anti-correlates with human ranks" finding, broadening the metric panel strengthens the claim.

---

## Recommended sequencing

**Tier 1 first.** It's the practical answer to "what does the user actually want?" — a desktop tool that produces good upscales without an API dependency, built almost entirely on what this study already shipped. 25 hours of engineering, no cloud spend, real shippable artifact. The capability study told us this is *feasible*; Tier 1 is the project that delivers it.

**Tiers 2–4 as a separate research track.** After Tier 1 ships, the SUPIR exploration tier becomes a different *kind* of project — research-flavored, longer, more uncertain in outcome, but pursuing the genuine SOTA gap. **Tier 2 (ZeroSFT) is the cheapest experiment that tells you the most** about whether the architecture lesson from this study generalizes. If Tier 2 lands the adapter in the "actually helpful" zone, Tier 3 becomes a defensible next step. If Tier 2 *also* fails, the conclusion shifts toward "diffusion-SR fundamentally needs research-scale data" and Tier 4 is the only path forward.

Don't skip Tier 2 to Tier 3/4. Tier 2 is the cheapest experiment that informs the bigger investments.

The two tracks (Tier 1 product + Tiers 2–4 research) can run in parallel if the engineering attention exists. They don't compete for the same hours; Tier 1 is plumbing-and-UX work, Tiers 2–4 are training-and-architecture work.
