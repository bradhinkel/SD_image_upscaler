# Follow-on project: closing the SUPIR gap incrementally

This document is the next-project pitch. Three tiers of ambition, each calibrated to a real time/cost budget. Each tier's value is tested by a clear empirical question that the previous tier didn't answer.

The 50-hour study (`SD_image_upscaler`) ended with a working pipeline that loses to Real-ESRGAN by ~40% LPIPS on clean bicubic LR and to SUPIR/HYPIR by another ~5–15%. Our four LoRA training attempts exposed *which* architectural choices matter most. Each tier below targets one or more of those.

---

## Tier 1 — ZeroSFT-style adapter, ~15 hrs, ~$10 cloud

**Core question: was the *adapter type* the issue?**

What changes vs current pipeline:
- Replace LoRA-on-cross-attention with a **zero-initialised additive adapter** on intermediate ResBlock features (after each ResBlock in stage B's SD 1.5 decoder). Same place SUPIR's ZeroSFT lives.
- Initial weights produce **exact zero delta** at training step 0 — base model behaves identically until training pushes the adapter weights, no destabilisation risk.
- Same 7,786-pair training data, same captions, same RunPod 5090 setup.

What stays:
- SD 1.5 stage-B base, ControlNet Tile structural conditioning, two-stage Phase 2 pipeline.
- BLIP-large captions (until Tier 2).
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
        # control is a same-spatial encoded image feature.
        scale = self.scale_proj(control)
        shift = self.shift_proj(control)
        return self.norm(x) * (1 + scale) + shift
```

Hook one ZeroSFT after each of stage B's down/up ResBlocks (~12 attachment points). Train alongside the existing pipeline; loss target same as current (MSE on predicted noise).

**Key empirical question:** does the same training recipe on a zero-init adapter produce a LoRA that *actually improves* the pipeline? Specifically, does it move the 5× LPIPS from ~0.43 (current two-stage) toward ~0.30 (Real-ESRGAN territory)?

**Risks / caveats:**
- The "control" input to ZeroSFT is non-trivial — SUPIR feeds it from the restoration encoder. Cleanest first try: use the LR image bicubic-upsampled to the latent's spatial size as control.
- ZeroSFT adds parameters proportional to UNet feature widths. ~5–10M trainable params at the small end (vs LoRA's 3M); still tractable on the 5090.

**Estimated effort:** 10 hrs eng (write ZeroSFT, integration plumbing, training script adaptation) + 1 hr cloud training (~$0.85) + 2 hrs eval / writeup. **Total ~15 hrs / $10.**

---

## Tier 2 — SDXL base + LSDIR-scale data + LLaVA captions, ~50 hrs, ~$50 cloud

**Core question: was the *base model + recipe + scale* the issue?**

What changes vs Tier 1:
- **Switch base from SD 1.5 to SDXL.** SUPIR's base; longer-trained, larger UNet, stronger prior.
- **Pull LSDIR proper.** Training data jumps from 2,600 sources → 50,000–100,000 sources, ~5× to 10× our current scale.
- **Caption with LLaVA** (or any larger VLM than BLIP-large) — SUPIR's specific Phase 4 finding.
- **Train longer**: 30,000–50,000 steps vs current 8,000.

What stays:
- ZeroSFT-style adapter from Tier 1 (now validated as the right architecture choice).
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

## Tier 3 — Full SUPIR-style restoration encoder, ~200 hrs, ~$300 cloud

**Core question: can we approach SUPIR with public tools at proper training scale?**

What changes vs Tier 2:
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

Independent of the tier choice, three cheap improvements would help any of them:

1. **Implement Phase 4.5 (prompt ladder).** The original plan had this and it got cut. Six prompt levels × 60 images = 360 inferences ≈ 6 GPU-hours total. The Phase 4.6 caption A/B was a 12-image one-engine slice; a full ladder on our two-stage pipeline would give a generalisable answer to "how much does prompting matter for our recipe."

2. **Wild-image test set.** Our 60-image set is bicubic-LR-only — exactly the regime where Real-ESRGAN dominates. A 30-image *real-world LR* set (phone photos, JPEG-recompressed images, low-light shots) would shift the comparison toward where SUPIR was actually designed to operate. Diffusion-SR's value proposition lives there, not on bicubic.

3. **A second metric for human-perception correlation.** Phase 3's correlation experiment used LPIPS, DISTS, PSNR, SSIM. Adding **DISTS-trained or Toolkit-MUSIQ** would expand the Phase 3 methodological finding. If the project is going to be cited for the "PSNR anti-correlates with human ranks" finding, broadening the metric panel strengthens the claim.

---

## Recommended sequencing

If picking one tier: **Tier 1.** It's the one that directly tests the architecture lesson from the failed 50-hour study. Cost is small ($10), time is two weekends, and the empirical answer materially informs whether Tier 2 is worth the bigger investment. If Tier 1 lands the LoRA in the "actually helpful" zone, Tier 2 becomes a defensible next step. If Tier 1 *also* fails, the conclusion shifts toward "diffusion-SR fundamentally needs research-scale data" and Tier 3 is the only path forward.

Don't skip to Tier 2/3. Tier 1 is the cheapest experiment that tells you the most.
