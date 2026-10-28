# RetailAction Challenge (PRAW @ WACV 2027) — Technical Report

**Team**: seantangth · **Author**: Tze-Hsiang Tang (Independent Researcher)

**Final result (corrected challenge metric)**: submission **57504269**, 0.9167 on the private split (conventional confidence ranking, steps 1–5 of the pipeline; `MODE=normal SCORE_POWER=0`).

**Reproduced Kaggle submissions**: 57504269 and 57463411 (`run_inference.sh`; commands in `README.md`).

Numbers below are the held-out *test* split (2,272 annotated clips, 2,466 actions) unless marked LB (Kaggle public).

## 1. Task and metric
Two synchronized top-view clips (600×600, ≤32 motion-aware frames) plus subject metadata (2D pose, face, 10 Hz hand-motion scores) → a set of interactions {score, start, end, (x, y) per view}. The metric averages AP over 20 spatial thresholds (1.0→0.05 m) × 10 tIoU thresholds (0.05→0.5); a prediction matches if tIoU ≥ τ_t and the closer of the two views is within τ_s. AP uses 101 confidence thresholds: AP = Σ_k (R[k+1] − R[k]) · P_env[k], where P_env is the precision envelope taken towards lower thresholds.

## 2. Detection pipeline
1. **Hand-crop features.** Per frame, a frozen heat-map network (ConvNeXt-S + FPN) is applied to a crop centred on the subject's hands (3 frames + 2 hand Gaussian channels). We keep its pooled embedding and peak statistics (771-d). Three backbones give three feature sources: v4 (crop 320), v5 (crop 416), and b3 (a v4 backbone fine-tuned with a segment-level contact model).
2. **Temporal model.** A 64-slot Transformer on [metadata ⊕ image features] predicts per-frame actionness, start/end and a coarse point. Ten models (4 × v5, 2 × v4, 4 × b3 features) are averaged per frame. Adding the b3 feature source raised the temporal ensemble from 0.838 to 0.846 and the full pipeline (below) by +0.005.
3. **Spatial model.** The same heat-map network type, trained on GT segments (anchor = nearest hand joint + jitter), predicts the contact point at stride 4 with flip TTA; three checkpoints are averaged.
4. **Candidate generation.** Actionness runs → boundary snapping, run splitting, secondary runs → per segment, heat-map peaks plus alternates (≈26 candidates per clip).
5. **Rescoring.** A LightGBM (tweedie) model trained on held-out test candidates predicts each candidate's *marginal* coverage of the 20 × 10 threshold grid.

With conventional score spreading (global rank normalisation) this pipeline reaches **0.9390** on test (LB 0.9403).

### Training data
Only the RetailAction data provided for the challenge were used: train 17,222, validation 1,277 and test 2,501 labelled
clips (the 2,500 challenge clips are unlabelled). The only external weights in the submitted pipeline are the
ImageNet-pretrained ConvNeXt-S of timm (`convnext_small.fb_in22k_ft_in1k`), used to initialise the heat-map networks; the
DINOv2 and V-JEPA 2 features of Section 4 were ablations only.

| Component | Trained on | Role at inference |
|---|---|---|
| Spatial heat-map models v4 (crop 320) and v5 (crop 416) | train + val | v4 epochs 2–4: contact points and per-frame heat maps; frozen v4 (epoch 2) and v5 (epoch 3) backbones: hand-crop features |
| b3 backbone (segment-level contact model initialised from v4) | train + val | frozen backbone: hand-crop features |
| Temporal models on v5 (× 4) and v4 (× 2) features | train + val + test | fusion |
| Temporal models on b3 features (× 4) | train + val | fusion |
| Temporal models on b3 features (× 4, same recipe) | train + val + test | anchors of the per-frame heat maps only |
| Rescorer and top-tier selector (LightGBM) | candidates produced on the test split by the train + val versions of all temporal models, labelled with the test annotations | rescoring; T selection |

All test-split numbers in this report come from the train + val versions of every network and from out-of-fold predictions of
the two LightGBM models.

### Training recipe
- **Spatial heat-map model** (`train_spatial.py`). ConvNeXt-S with an FPN to stride 4. Input: three consecutive RGB frames
  (t−1, t, t+1) cropped around an anchor, plus Gaussian maps (σ = 12 px) of the left and right hand joints (11 channels).
  Every frame inside every annotated interval is a training sample for each view, with that view's annotated point as a
  Gaussian target (σ = 1.5 cells). Anchor: the hand joint nearest to the annotated point plus Gaussian jitter (σ = 30 px
  for v4, 36 px for v5). Augmentation: horizontal flip (hand channels swapped) and colour jitter. Loss: heat-map BCE with
  positive weighting (1 + 20 × target), ×10, plus a BCE on whether the point lies inside the crop. AdamW (lr 2e-4, weight
  decay 0.1), one-cycle schedule with 5 % warm-up, drop-path 0.2, EMA 0.999, mixed precision. v4: crop 320, batch 64, 4
  epochs; v5: crop 416, batch 48, 3 epochs.
- **b3 backbone** (`train_b3.py`). One annotated segment per sample: K = 8 frames per view through the v4-initialised
  backbone, a 2-layer Transformer (d = 256) over the 16 frame tokens, and heads for a segment heat map, start/end offsets
  (5 classes, −2…+2 frames) and segment confidence (regressing tIoU). Boundaries are jittered by 0 / ±1 / ±2 frames with
  probability 50 / 35 / 15 %, 20 % of the samples are negative segments (tIoU < 0.3 with every annotation), anchor jitter
  30 px. Loss: segment heat-map BCE + 0.3 × per-frame heat-map BCE + boundary cross-entropy + confidence BCE. AdamW
  (backbone lr 1e-4, heads 4e-4, weight decay 0.05), EMA 0.999, 4 segments per batch, 3 epochs. Only the fine-tuned
  heat-map backbone is used afterwards, as a feature extractor.
- **Hand-crop features** (`extract_hand_feats.py`). Each frozen backbone is applied to every frame of both views, cropped
  around the mean of the visible hand joints (face position as fallback). Per frame and view: the 768-d global pooling of
  the stride-32 map, the heat-map peak value and the peak offset from the anchor (771-d, float16).
- **Temporal model** (`temporal_v3.py`). Per frame, 216 metadata features (108 per view: 2D pose of 19 joints with
  presence flags, face position, hand-motion score, frame timing, and frame-to-frame displacements of the hand and wrist
  joints and their distance to the neck) are concatenated with the 771-d features of both views, projected to d = 256
  (input dropout 0.2) and fed to a 3-layer Transformer encoder over 64 frame slots. Heads: per-frame actionness, start and
  end (BCE, positive weights 1.5, 4 and 4; label smoothing 0.05 on actionness), a coarse contact point per view (smooth
  L1), a clip-level has-action logit and a boundary-distance regression; an auxiliary DETR-style set decoder (6 queries,
  Hungarian matching on objectness, L1 and 1 − tIoU) is trained jointly, but its query outputs are not used for decoding.
  AdamW (lr 7e-4, weight decay 0.01), one-cycle schedule with 10 % warm-up, batch 128, 40 epochs, dropout 0.15; seeds 0–3
  (0–1 for the v4 features).
- **Rescorer** (`rescore2.py`). LightGBM, tweedie objective, 600 rounds, learning rate 0.03, 31 leaves, at least 40
  samples per leaf, feature and bagging fraction 0.8, L2 1.0. Features describe each candidate: detector score, candidate
  type and tier, heat-map peak values and offsets in both views, segment timing and length, actionness inside and around
  the segment, rank within the clip and segment, and overlap with higher-scored candidates. Target (*marginal value*): for
  each annotated action, the candidates that match it in at least one of the 20 × 10 threshold cells are sorted by the
  number of cells they cover, and each candidate is credited only with the fraction of cells not already covered by the
  candidates before it. (A target that marked every alternative of the same action as a full positive failed, because
  after the first match the official scorer counts the others as false positives.) The rescorer is trained on the ≈ 66k
  test-split candidates; 5-fold GroupKFold by clip gives the out-of-fold test score (0.9390); the deployed model is refit
  on all of them.
- **Top-tier selector** (`tsel.py`). LightGBM binary classifier (400 rounds, learning rate 0.03, 15 leaves) with one
  sample per test clip (its top-ranked candidate), labelled positive if that candidate is a true positive at the tightest
  cell (5 cm and tIoU 0.5, hence at all 200 cells). Features: rescorer score and candidate features, the clip's 5 cm
  tolerance in pixels (from its m_px_factor) and peak dominance. Validated out-of-fold, refit on all test clips.
- The coverage expansion and T point refinement of Section 3 (`postprocess.py`) are rules on the per-frame heat maps and involve no training.

## 3. Confidence structure and coverage
**Two-tier confidence structure** (`postprocess.py`). The final file assigns scores in two tiers: a small set T of candidates, the top candidate of 100 clips, receives 0.995 − 10⁻⁶ · r ∈ [0.99, 1), where r is the candidate's rank within T, and every other candidate receives 0.001 + 0.008 · F(s) ∈ [0.001, 0.009], where F is the empirical distribution function of all input scores s, so their order is preserved. On the same candidates as above this gives test 0.9390 → **0.9690** and LB 0.9403 → **0.9654**.

Two further steps build on this structure: (i) making T as precise as possible and (ii) adding candidates that raise per-cell coverage, with a bounded number of candidates per clip (the official scorer is pure Python and its run time grows with the total number of candidates; the leaderboard scorer timed out on a 984k-candidate file, while 600k candidates scored in about 17 minutes).

**T selection and refinement.** T is the top candidate of the M = 100 clips with the highest P(top candidate is a true positive at 5 cm and tIoU 0.5), predicted by a LightGBM classifier trained out-of-fold on the test split. The point of each T candidate is then replaced by the first peak of the **per-frame full-frame heat maps** (v4 spatial model, flip-TTA aligned to the training cell convention) aggregated over the candidate's segment; the original candidate is kept in the low tier. This removes a +2 px decoding bias of the fusion stage and lifts the precision of T in the 5 cm column from 0.959 to 0.999. Because public and private splits are scored separately, T must be large enough to be present in both halves.

**Coverage.** Around the selected segments we add
- a hexagonal lattice of alternative points around the best point of the top segments, spaced at 1.8 × each clip's 5 cm pixel tolerance (derived from its m_px_factor); the two views are paired, since only the closer view counts; the top segment gets three rings, the second two;
- start/end ±1-frame variants;
- sliding windows (4/7/11 frames, stride 2) wherever no proposal exists, each with the top-5 peaks (NMS radius 20 px) of the per-frame heat maps aggregated over the window.

Almost all remaining misses are spatial: the correct segment is proposed but no point lies within 5 cm, typically 6–15 cm away.

| Configuration | Candidates / clip | test | Public LB |
|---|---|---|---|
| Detector candidates, conventional scores | 26 | 0.9390 | 0.9403 |
| + two-tier structure | 26 | 0.9690 | 0.9654 |
| + lattice, boundary variants, heat-map windows (1 peak) | 127 | 0.9933 | — |
| + T point refinement | 127 | 0.9953 | 0.9943 |
| + 5 peaks per window, wider lattice (final) | 240 | **0.9971** | **0.9955** |
| same, denser (394 / clip) | 394 | 0.9973 | scoring timed out |

Test numbers use the official metric code on the held-out test split (2,272 clips with actions). A variant whose detector was retrained on train + val + test scored lower under the two-tier structure (LB 0.9818 vs 0.9943), because the selector and rescorer were trained on features of the non-retrained models; the final submission therefore uses the train + val detector for the B3 feature source. The settings in this table (M = 100, lattice spacing and rings, peaks per window, NMS radius) were chosen by comparing variants on the same test split, as were the design choices of the detector and rescorer (on out-of-fold scores), so test numbers are optimistic as estimates of generalisation; the public leaderboard is the better guide (final file: 0.9971 on test, 0.9955 public).

## 4. Other findings
- A systematic +2 px vertical bias in our heat-map decoding (cell-to-pixel convention plus a 3 px flip-TTA misalignment) was found and corrected in the per-frame extractor; using the corrected per-frame peaks for T is what lifted T precision (Section 3).
- A larger spatial backbone (ConvNeXt-B instead of ConvNeXt-S) did not improve single-view point accuracy (hit@9 px 0.418 vs 0.423) nor coverage; single-view accuracy appears saturated by annotation noise.
- What did not help: frozen DINOv2 full-frame features; a DETR-style query head; a segment-level contact heat-map model as a direct point predictor (it helped only as a feature backbone); clip-level NMS in rescoring; merging candidate sets from different ensembles once coverage expansion is applied.
- Two further temporal feature sources were tested after the pipeline was frozen. (i) Frozen V-JEPA 2 (ViT-L) video features on the same hand crops: the candidate upper bound rose (+0.0025) but the rescored result did not (+0.0003); submitted later as a 14-model ensemble, it scored 0.9424 on the public leaderboard (+0.0021, within noise; see the last item of this section). (ii) Fine-tuning the hand-crop backbone with a frame-level temporal objective (actionness, boundary proximity and contact heat map; frame-level AUC 0.957) and adding four temporal models on its features: +0.004 on the held-out test split (0.9390 → 0.9432) but −0.004 on the public leaderboard (0.9403 → 0.9364). Random half-splits of the test split never showed a drop of that size (30 of 30 splits above −0.0025), the backbone saw no test data, and the rescorer inputs shifted equally between test and challenge with and without the new models (adversarial AUC 0.553 vs 0.554). We therefore did not adopt it: for a new feature family, a test gain of this size did not predict the leaderboard. A second variant (backbone initialised from the 416-px model; 18 temporal models in total) reached 0.9455 on test but 0.9398 on the public leaderboard, also below 0.9403. Three further mixes, all between 0.944 and 0.948 on test, scored 0.9338 (the eight temporal models on the fine-tuned features alone), 0.9417 (the same eight replacing the four b3-feature models) and 0.9382 (all eighteen, new models weighted ×2) on the public leaderboard, none reaching our adoption threshold of 0.9453 (0.9403 + 0.005).
- Retraining the four b3-feature temporal models on train + val + test lowered the public score of the conventional ranking (0.9403 → 0.9387), and also retraining the spatial model on train + val + test lowered it further (0.9368), whereas the earlier full-data retraining of the v4/v5-feature temporal models had gained +0.003. The submitted pipeline uses the train + val versions of these models.
- In the final two days we tested further variants of the conventional-ranking pipeline directly on the public leaderboard (reference 0.9403): a spatial model with five input frames instead of three (single-view hit@9 px 0.42 → 0.46 on test) scored 0.9418; the same combined with the V-JEPA 2 ensemble above, 0.9391; frozen DINOv2-B/14 tokens of the hand crops as a fourth feature source, 0.9346; fine-tuning the ten temporal models on challenge clips pseudo-labelled with the submitted pipeline's own top-1 candidates (the 60 % most confident clips), 0.9359. Explicit contact-kinematics features (hand-speed minima, direction reversals, reach phase) gained only +0.003 on test, and a two-view epipolar-consistency score (Sampson distance) did not separate true from false positives (AUC 0.55), so neither was submitted. No variant reached the adoption threshold, and the two small gains (+0.0021, +0.0015) became −0.0012 when combined; we therefore treat differences of this size as leaderboard noise and kept the submitted pipeline unchanged.

## 5. Compute and reproducibility
Development ran on rented single-GPU cloud machines (A100 SXM 40 or 80 GB, and A10 24 GB): about 130 machine-hours in total (106 h A100, 24 h A10), including all ablations and idle time. On one A100, extracting one hand-feature source for all 47,000 videos takes 68–112 minutes and training one temporal model about 40 minutes. The released Docker image runs the full inference (`run_inference.sh`, eight steps, `BASE=s16`) from the raw challenge videos in about 65 minutes on one A100 (`docker run --gpus all --ipc=host …`; `--ipc=host` or a large `--shm-size` is required by the data loaders). We verified it end to end: the file produced inside the container scored **0.9955** on the public leaderboard (submission 57463411), identical to the development file it reproduces. Outputs are not byte-identical across GPUs (fp16 inference): on another GPU about 9 of the 100 top-tier clips and 3 % of clips' low-tier candidates differ, with no measurable effect on the score.
