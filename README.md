# RetailAction Challenge (PRAW @ WACV 2027) — inference code, team *seantangth*

This repository reproduces our Kaggle submission **57463411** (public leaderboard **0.9955**) from the raw challenge videos
with a single Docker command.
It contains inference code only; the method is described in [`REPORT.md`](REPORT.md).

## Quick start

```bash
git clone https://github.com/seantangth/WACV2027-RetailAction-Challenge.git && cd WACV2027-RetailAction-Challenge
bash download_weights.sh              # ~1.2 GB of checkpoints from Hugging Face into ./weights
docker build -t rac-infer .
docker run --gpus all --ipc=host --rm \
  -v /path/to/kaggle_data:/data:ro \
  -v "$PWD/out":/out \
  rac-infer
# -> out/submission.csv  (2,500 rows, Kaggle submission format)
```

* `/data` is the Kaggle competition data root. It must contain `sample_submission.csv` and
  `challenge_test_set/vam-dataset-challenge_test_no_labels/data/<id>/{rank0_video.mp4, rank1_video.mp4, metadata.json}`.
* `--ipc=host` (or `--shm-size=64g`) is **required**: the data loaders pass multi-frame crops through shared memory and
  fail with a bus error under Docker's default shared-memory size.
* Hardware used for verification: 1× NVIDIA A100 40 GB, 30 vCPUs, 200 GB RAM. Runtime ≈ 65 minutes
  (features 28 min, temporal models 3 min, heat-map fusion 15 min, per-frame heat maps 12 min, rest < 5 min).
  Only a 40 GB A100 was tested; `nvidia-smi` showed up to 37 GB in use during heat-map fusion (batch size 16).
  On a GPU with less memory, lower the `--bs` values in `run_inference.sh` (step 4 heat-map fusion: 16; step 2
  hand-crop features: 4; step 6 per-frame heat maps: 2). This is untested, and fp16 results may then differ slightly.

Optional environment variables (`docker run -e NAME=value`):

| Variable | Default | Meaning |
|---|---|---|
| `WORKERS` | 8 | data-loader workers per step (16 used for verification) |
| `START_STEP` | 1 | resume from step *k* (outputs of earlier steps must already be in `/out`) |
| `AUG` | final configuration | JSON of the coverage-expansion settings used in step 8 |

## Pipeline (`run_inference.sh`)

1. **Metadata parsing** — 2D pose, face box, hand-motion scores, frame timestamps; per-view metre-per-pixel factor.
2. **Hand-crop features** — three frozen heat-map backbones (ConvNeXt-S + FPN) applied to per-frame crops around the hands.
3. **Temporal models** — 14 Transformers (predict-only) producing per-frame actionness, boundaries and a coarse point.
4. **Heat-map fusion** — segments from the temporal ensemble, contact points from the spatial heat-map model (flip TTA), candidate expansion.
5. **Rescoring** — LightGBM (tweedie) predicting each candidate's marginal coverage of the metric's threshold grid.
6. **Per-frame heat maps** — full-frame contact heat maps for every frame (used by steps 7–8).
7. **Top-tier selector** — LightGBM classifier choosing the 100 clips whose top candidate is most likely a true positive at the tightest threshold.
8. **Coverage expansion and confidence structure** — lattice points, boundary variants and sliding windows with heat-map peaks; the 100 selected candidates (points refined to the per-frame heat-map peak) receive scores in [0.99, 1) and all other candidates scores in [0.001, 0.009].

## Reproducibility

* The file produced by this container on an A100 scored **0.9955** on the public leaderboard (submission 57463411), identical to the
  development file it reproduces.
* Inference runs in fp16 on GPU, so results are not bit-identical across GPU models: on another GPU about 9 of the 100
  top-tier clips and ~3 % of clips' low-tier candidates differed, with no measurable change in score.
* The official scorer's run time grows with the number of candidates; the final file has ~600k candidates (≈ 17 min to score on Kaggle).

## Weights

`weights/MANIFEST.txt` lists the 40 files (spatial backbones, temporal models, rescorer and selector). They are hosted at
Hugging Face `seantangth/WACV2027-RetailAction-Challenge-weights` and fetched by `download_weights.sh`.

## License and attribution

Code: MIT (see `LICENSE`). The model weights were trained on the RetailAction dataset by Standard Cognition (Standard AI),
used under its dataset license; see `NOTICE`. The dataset is not redistributed here.
