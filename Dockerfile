FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir av scipy requests
COPY . .
# Mount the Kaggle download at /data, the two checkpoints (base.pt and
# refined.pt) at /ckpt and an output directory at /out, for example:
# docker run --gpus all --shm-size=8g -v $PWD/data:/data -v $PWD/ckpt:/ckpt -v $PWD/out:/out retailaction
CMD ["./run_inference.sh"]
