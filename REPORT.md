# RetailAction challenge: method report

Team MaxwellEQ (Kaggle display name Arthur12212003). Member: Chi-Wei Lee, National Tsing Hua University, Taiwan.
Best public score: 0.7939 (submission 56635294, from a checkpoint we could not keep). Reproducible final submission: 0.7924 (submission 57249716), the average of two retrained checkpoints described below.

## Summary

We re-implemented the organisers' MViT-B + DETR baseline, trained it on all labelled clips, and added three changes at inference and fine-tuning: foreground rescoring, horizontal flip test-time augmentation with query pairing, and a stride-8 point refinement step. Everything uses the competition data and public Kinetics-400 weights only.

## Model

- Input: 16 frames at 224x224 from each of the two camera views.
- Backbone: one MViT-B shared by both views (torchvision Kinetics-400 weights). Tokens are projected to 256 dimensions, given learned 3D positional encodings, and passed through one transformer encoder per view.
- Decoder: the two token sets are concatenated with a view embedding and decoded by a 6 layer DETR decoder with 5 action queries. Each query predicts class logits (take, put, touch, none), a normalized interval, and a touch point in each view.
- Training: Hungarian matching on class probability, point distance and temporal L1 / IoU. Losses are cross entropy, point L2, temporal L1 and 1 - tIoU, with auxiliary losses on every decoder layer. Effective batch 32, learning rate 1e-4 for the head and 2e-5 for the backbone, 14 epochs.
- Data: all 17,222 labelled training clips plus the labelled split of the HuggingFace test set. The HuggingFace validation split was held out for model selection and all ablations below.

## Inference

1. Foreground rescoring: confidence = 1 - p(no action), instead of the highest action class probability.
2. Flip TTA: the clip is also run mirrored. The two sets of queries are paired with the Hungarian algorithm, and times and points are averaged (points un-mirrored first). Confidence is averaged.
3. Point refinement (optional, used in a second model): two correction steps sample a 3x3 patch of stride-8 backbone features around each predicted point at the query's temporal centre. The correction is zero-initialised, so the model starts from the base checkpoint and is fine-tuned for 3 epochs at lr 5e-5 / 1e-5.

## Results

Validation mAP uses the lower of the two per-view scores (our approximation of the leaderboard rule). The mean of the two views is about 0.05 lower in every row.

| Model | Score | TTA | Val mAP | Public |
|---|---|---|---|---|
| 8k clip subset, first run | max | none | 0.6940 | 0.7567 |
| 8k clip subset, final checkpoint | max | none | 0.7448 | |
| 8k clip subset, final checkpoint | max | flip | 0.7550 | 0.7676 |
| 8k subset + refinement | max | flip | 0.7717 | 0.7869 |
| Full data | max | none | 0.7518 | |
| Full data | max | flip | 0.7573 | |
| Full data | fg | none | 0.7578 | |
| Full data | fg | flip | 0.7704 | 0.7939 |
| Full data, retrained (seed differs) | fg | flip | 0.7656 | |
| Retrained + refinement | fg | flip | 0.7666 | |
| Average of the two retrained models | fg | flip | 0.7722 | 0.7924 |

The 0.7939 checkpoint was deleted by a disk clean-up, so we retrained the same recipe. The retrain is 0.005 lower on validation, which is the size of the seed variation. Averaging it with its refined version (queries paired per clip with the Hungarian algorithm, probabilities, times and points averaged) recovers 0.7722 on validation and 0.7924 on the public board. This ensemble is what the Docker image reproduces.

Foreground rescoring helped the full-data checkpoint (+0.013 with flip) and hurt the 8k checkpoint, so it is chosen per checkpoint on validation. Refinement gave +0.017 validation and +0.019 public on the 8k base.

## Compute

One RTX 5090. The full-data base model trains in about 4.6 GPU hours, refinement in about 3, and test inference with both passes takes a few minutes.

## Reproduction

The repository holds inference-only code and a Dockerfile, and the two checkpoints (`base.pt`, `refined.pt`) are attached to its v1.0 release. With the Kaggle zip in `data/` and the checkpoints in `ckpt/`, one command reproduces submission 57249716:

    docker run --rm --gpus all --shm-size=8g -v $PWD/data:/data -v $PWD/ckpt:/ckpt -v $PWD/out:/out retailaction

The container packs the test clips, dumps the raw query outputs of both checkpoints for the plain and flipped passes, folds the flip pass, averages the two models, applies foreground rescoring and writes `/out/submission.csv` after a format check. The README lists every step, including the download commands and the expected checksum of the output.
