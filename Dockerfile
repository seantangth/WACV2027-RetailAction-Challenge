# RetailAction Challenge (PRAW @ WACV 2027) — inference-only image (S28 pipeline: 3 feature sources, 10 temporal models, heatmap fusion, LightGBM rescorer, per-frame heatmaps, top-tier selector + refinement, coverage expansion, two-tier scores)
# Build:  docker build -t rac-infer .
# Run:    docker run --gpus all --rm --ipc=host -v /path/to/kaggle_data_root:/data:ro -v $PWD/out:/out rac-infer
#   --ipc=host (or --shm-size=64g) is required: data loaders pass large multi-frame crops through shared memory.
#   /data must contain challenge_test_set/vam-dataset-challenge_test_no_labels/data/<id>/{rank0_video.mp4,rank1_video.mp4,metadata.json} and sample_submission.csv
#   Run download_weights.sh before building (weights/ is copied into the image).
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 RAC_NO_PRETRAINED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 timm==1.0.19 av==15.0.0 lightgbm==4.7.0 scikit-learn==1.6.1 huggingface_hub
WORKDIR /app
COPY src/ /app/src/
COPY weights/ /app/weights/
COPY run_inference.sh /app/run_inference.sh
RUN chmod +x /app/run_inference.sh
ENTRYPOINT ["/app/run_inference.sh"]
