# The test clips are re-encoded with ffmpeg before inference, and the
# predictions depend on the exact encoder build, so this uses the ffmpeg 6.1
# of Ubuntu 24.04 that produced the submission, from a frozen package snapshot.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
ARG SNAPSHOT=20261016T000000Z
# The snapshot is served over HTTPS, so certificates come first.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get update --snapshot $SNAPSHOT \
    && apt-get install -y --no-install-recommends --snapshot $SNAPSHOT ffmpeg python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# Pinned to the versions the final submission was produced with.
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir numpy==2.2.5 av==18.1.0 scipy==1.17.1 requests==2.32.3
WORKDIR /app
COPY . .
# Mount the Kaggle download at /data, the two checkpoints (base.pt and
# refined.pt) at /ckpt and an output directory at /out, for example:
# docker run --rm --gpus all --shm-size=8g -v $PWD/data:/data -v $PWD/ckpt:/ckpt -v $PWD/out:/out retailaction
CMD ["sh", "run_inference.sh"]
