#!/usr/bin/env bash
# RetailAction Challenge — reproduce the final submission with one command: /data (Kaggle data root) -> /out/submission.csv
# Steps: 1 metadata -> 2 hand-crop features (3 frozen backbones) -> 3 temporal models x14 (predict-only)
#        -> 4 heat-map fusion + candidates -> 5 LightGBM rescorer -> 6 per-frame heat maps -> 7 top-tier selector
#        -> 8 coverage expansion + two-tier confidence structure + top-tier point refinement -> submission.csv
# Env: DATA_ROOT (/data), OUT_DIR (/out), WORKERS (8), START_STEP (1), AUG (coverage JSON), BASE (s16 = final submission),
#      WEIGHTS_DIR / SRC_DIR (override for native runs outside Docker),
#      MODE (twotier = steps 1-8, default; normal = steps 1-5 only: plain confidence ranking, score = rescorer prediction ^ SCORE_POWER),
#      SCORE_POWER (2; MODE=normal only).
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/data}; OUT_DIR=${OUT_DIR:-/out}; WK=${WORKERS:-8}; W=${WEIGHTS_DIR:-/app/weights}; S=${SRC_DIR:-/app/src}
MODE=${MODE:-twotier}; SCORE_POWER=${SCORE_POWER:-2}
case "$MODE" in twotier|normal) ;; *) echo "MODE must be twotier or normal (got: $MODE)" >&2; exit 2;; esac
BASE=${BASE:-s16}   # s16: final submission (B3 temporal models trained on train+val for fusion; full-data models as heat-map anchors)
mkdir -p "$OUT_DIR/meta" "$OUT_DIR/raw" "$OUT_DIR/feats_v4" "$OUT_DIR/feats_v5" "$OUT_DIR/feats_b3" "$OUT_DIR/frame_hm"; cd "$S"
SS="$DATA_ROOT/sample_submission.csv"; [ -f "$SS" ] || SS="$DATA_ROOT/challenge_test_set/sample_submission.csv"
M_V5="tv4v5f_s0 tv4v5f_s1 tv4v5f_s2 tv4v5f_s3"; M_V4="tv4f_s0 tv4f_s1"; M_B3="tvb3f_s0 tvb3f_s1 tvb3f_s2 tvb3f_s3"; M_B3NF="tvb3_s0 tvb3_s1 tvb3_s2 tvb3_s3"
TV="--d 256 --layers 3 --in_drop 0.2 --drop 0.15 --predict_split challenge"
R=""; for m in $M_V5 $M_V4 $M_B3; do R="$R $OUT_DIR/raw/$m/raw_challenge.pkl"; done          # per-frame heat-map anchors: full-data temporal models x10
RF="$R"; [ "$BASE" = "s16" ] && { RF=""; for m in $M_V5 $M_V4 $M_B3NF; do RF="$RF $OUT_DIR/raw/$m/raw_challenge.pkl"; done; }   # temporal models used for fusion
CK="$W/spatial_v4/model_ep2.pt $W/spatial_v4/model_ep3.pt $W/spatial_v4/model_ep4.pt"                        # per-frame heat maps
CKF="$W/spatial_v4f/model_ep2.pt $W/spatial_v4f/model_ep3.pt $W/spatial_v4f/model_ep4.pt"; [ "$BASE" = "s16" ] && CKF="$CK"   # fusion
# AUG: coverage-expansion settings (final: ~240 candidates per clip); --trefine moves top-tier points to per-frame heat-map peaks
AUG=${AUG:-'{"k_lat":2,"lat_rings":2,"lat_rings_top1":3,"k_tv":4,"tv_r":1,"win":[4,7,11],"fhm_win":1,"fhm_win_k":5,"fhm_win_nms":5,"lat_c":1.8}'}
START_STEP=${START_STEP:-1}; step(){ [ "$START_STEP" -le "$1" ]; }   # resume from step k (earlier outputs must exist in OUT_DIR)
if step 1; then
echo "[1/8] parse metadata"
python parse_meta.py --root "$DATA_ROOT" --out "$OUT_DIR/meta" --splits challenge --workers "$WK"
fi
if step 2; then
echo "[2/8] hand-crop per-frame features (frozen backbones v4 crop320, v5 crop416, b3 crop320)"
python extract_hand_feats.py --meta_dir "$OUT_DIR/meta" --raw "$DATA_ROOT" --ckpt "$W/spatial_v4/model_ep2.pt" --out "$OUT_DIR/feats_v4" --splits challenge --workers "$WK" --bs 4
python extract_hand_feats.py --meta_dir "$OUT_DIR/meta" --raw "$DATA_ROOT" --ckpt "$W/spatial_v5/model_ep3.pt" --out "$OUT_DIR/feats_v5" --splits challenge --workers "$WK" --bs 4
python extract_hand_feats.py --meta_dir "$OUT_DIR/meta" --raw "$DATA_ROOT" --ckpt "$W/b3_v1_heat/model.pt" --out "$OUT_DIR/feats_b3" --splits challenge --workers "$WK" --bs 4
fi
if step 3; then
echo "[3/8] temporal models (10x, predict-only)"
for m in $M_V5; do python temporal_v3.py --meta_dir "$OUT_DIR/meta" --out_dir "$OUT_DIR/raw/$m" --img_feats "$OUT_DIR/feats_v5" --predict_only "$W/temporal/$m/model.pt" $TV; done
for m in $M_V4; do python temporal_v3.py --meta_dir "$OUT_DIR/meta" --out_dir "$OUT_DIR/raw/$m" --img_feats "$OUT_DIR/feats_v4" --predict_only "$W/temporal/$m/model.pt" $TV; done
for m in $M_B3; do python temporal_v3.py --meta_dir "$OUT_DIR/meta" --out_dir "$OUT_DIR/raw/$m" --img_feats "$OUT_DIR/feats_b3" --predict_only "$W/temporal/$m/model.pt" $TV; done
[ "$BASE" = "s16" ] && for m in $M_B3NF; do python temporal_v3.py --meta_dir "$OUT_DIR/meta" --out_dir "$OUT_DIR/raw/$m" --img_feats "$OUT_DIR/feats_b3" --predict_only "$W/temporal/$m/model.pt" $TV; done
fi
if step 4; then
echo "[4/8] spatial heatmap fusion + candidate expansion"
python fuse_predict.py --meta_dir "$OUT_DIR/meta" --raw_root "$DATA_ROOT" --split challenge --temporal_raw $RF --spatial_ckpt $CKF \
  --k_runs 3 --thr 0.4 --ranknorm --snap --tta --multi --thr2 0.2 --bvar --alt_secondary --workers "$WK" --bs 16 \
  --write "$OUT_DIR/fused.csv" --save_feats "$OUT_DIR/fused_feats.pkl" --sample_sub "$SS"
fi
if step 5; then
echo "[5/8] second-stage rescoring (LightGBM tweedie, pre-trained)"
python rescore2.py --meta_dir "$OUT_DIR/meta" --csv x --feats x --load_model "$W/rescorer/rescorer_final" \
  --apply_csv "$OUT_DIR/fused.csv" --apply_feats "$OUT_DIR/fused_feats.pkl" --sample_sub "$SS" --out "$OUT_DIR/rescored.csv"
if [ "$MODE" = "normal" ]; then
  # plain confidence ranking: same candidates and order as rescored.csv; score = predicted marginal value ^ SCORE_POWER (monotonic)
  python rescore2.py --meta_dir "$OUT_DIR/meta" --csv x --feats x --load_model "$W/rescorer/rescorer_final" --score_power "$SCORE_POWER" \
    --apply_csv "$OUT_DIR/fused.csv" --apply_feats "$OUT_DIR/fused_feats.pkl" --sample_sub "$SS" --out "$OUT_DIR/submission.csv"
fi
fi
if [ "$MODE" = "normal" ]; then echo "done (MODE=normal, SCORE_POWER=$SCORE_POWER) -> $OUT_DIR/submission.csv"; exit 0; fi
if step 6; then
echo "[6/8] per-frame full-frame heatmaps (v4 ep2-4, aligned flip TTA)"
python frame_hm.py --meta_dir "$OUT_DIR/meta" --raw_root "$DATA_ROOT" --split challenge --temporal_raw $R --spatial_ckpt $CK --out "$OUT_DIR/frame_hm" --workers "$WK" --bs 2
fi
if step 7; then
echo "[7/8] top-candidate selector (P(TP at 5cm & tIoU0.5))"
python tsel.py apply --meta_dir "$OUT_DIR/meta" --split challenge --csv "$OUT_DIR/rescored.csv" --feats "$OUT_DIR/fused_feats.pkl" --model "$W/rescorer/tsel_final" --out "$OUT_DIR/tsel.json" --M 100
fi
if step 8; then
echo "[8/8] coverage expansion + two-tier confidence structure + top-tier point refinement -> submission.csv"
python postprocess.py apply --meta_dir "$OUT_DIR/meta" --split challenge --csv "$OUT_DIR/rescored.csv" --fhm "$OUT_DIR/frame_hm" \
  --aug2 "$AUG" --tsel "$OUT_DIR/tsel.json" --trefine --sample_sub "$SS" --out "$OUT_DIR/submission.csv"
fi
echo "done -> $OUT_DIR/submission.csv"
