FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir av scipy requests
COPY . .
# Mount the Kaggle download at /data, the trained checkpoint at /ckpt/best.pt
# and an output directory at /out, for example:
# docker run --gpus all -v $PWD/data:/data -v $PWD/ckpt:/ckpt -v $PWD/out:/out retailaction
# SCORE is "fg" for the base checkpoint and "max" for the refined one.
ENV SCORE=fg
CMD python -m retailaction.pack --src /data/retail-action-challenge-praw-workshop-wacv-2027.zip \
        --prefix challenge_test_set/ --out /out/challenge.pack.tar && \
    python -m retailaction.dump --ckpt /ckpt/best.pt --pack /out/challenge.pack.tar --out /out/test_raw.npz && \
    python -m retailaction.post --raw /out/test_raw.npz --score $SCORE --tta flip \
        --sample /data/sample_submission.csv --out /out/submission.csv && \
    python -m retailaction.check_submission /out/submission.csv --sample /data/sample_submission.csv
