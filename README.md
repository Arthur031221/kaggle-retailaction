# RetailAction challenge (PRAW workshop, WACV 2027)

Code for the Kaggle competition
[retail-action-challenge-praw-workshop-wacv-2027](https://www.kaggle.com/competitions/retail-action-challenge-praw-workshop-wacv-2027).

## Task

Each sample is a pair of short (at most 32 frame) top-view clips of one shopper,
seen from two synchronized ceiling cameras (rank0 and rank1). For every
shopper-shelf interaction the model outputs a confidence, a normalized start and
end time, and a normalized (x, y) touch point in each view. The metric is mAP
averaged over temporal IoU thresholds (0.05 to 0.5), spatial distance
thresholds in meters (1.0 to 0.05) and confidence thresholds. The action class
(take, put, touch) is not scored.

Data: the Kaggle competition files and the RetailAction dataset
(`standard-cognition/RetailAction` on HuggingFace, Standard AI dataset licence).
No data is included in this repository.

## Approaches

1. **Pose rule** (`retailaction/poserule.py`, no training). The metadata
   carries 2D keypoints of the shopper for every frame. The frame and hand with
   the largest arm extension (hand to neck distance over torso length, summed
   over both views) is taken as the contact; the hand keypoint in each view is
   the touch point; the interval is a blend of a fixed width window and the
   span where extension stays near its peak. A few weaker peaks are added with
   lower scores. The handful of parameters in `configs/poserule.json` were
   tuned on half of the validation split. Note: keypoints are stored as
   [row, col], labels as (x, y).
2. **MViT-B + DETR baseline** (`retailaction/model.py`), a reimplementation of
   the organisers' baseline from the RetailAction paper (ICCV 2025 Retail
   Vision workshop). A shared MViT-B backbone (Kinetics-400 weights from
   torchvision) encodes 16 frames at 224x224 from each view; tokens are
   projected to 256 dims, given learned 3D positional encodings and refined
   by one transformer encoder per view; the two token sets are concatenated
   with a view embedding and decoded by a 6 layer DETR decoder with 5 action
   queries. Each query predicts class logits (take, put, touch, none), an
   interval and a point per view. Hungarian matching uses class probability,
   point distance and temporal L1 / IoU; losses are cross entropy, point L2,
   temporal L1 and 1 - tIoU, with auxiliary losses on every decoder layer.
   The confidence is the highest action class probability.

   Differences from the paper: effective batch 32 instead of 1024 with a lower
   backbone learning rate (2e-5, head 1e-4), a 1 - tIoU loss term, and training
   on a random subset of 8,000 of the 17,222 training clips because of local
   disk limits.

## Layout

```
retailaction/pack.py        re-encode clips to 256x256 and pack them into one tar + index
retailaction/remote.py      seekable HTTP file (range requests) for remote zips and tars
retailaction/data.py        dataset over packed tars (PyAV decoding, crop / flip augmentation)
retailaction/model.py       MViT-B + DETR head, matcher and losses
retailaction/metric.py      local approximation of the challenge metric
retailaction/poserule.py    pose rule and its submission writer
retailaction/train.py       training loop with validation
retailaction/infer.py       checkpoint -> submission CSV
retailaction/dump.py        raw query outputs of a checkpoint, plain and horizontally flipped
retailaction/post.py        scoring and flip averaging over dumped outputs; evaluation or CSV
retailaction/check_submission.py   format check against sample_submission.csv
tools/overlap_check.py      checks that no challenge test clip appears in a labelled split
configs/poserule.json       tuned pose rule parameters
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # needs ffmpeg on PATH

# 1. pack the challenge test set (from the Kaggle download, a directory or a zip)
python -m retailaction.pack --src retail-action-challenge-praw-workshop-wacv-2027.zip \
    --prefix challenge_test_set/ --out data/challenge.pack.tar

# 2a. pose rule submission (CPU only, about a second)
python -m retailaction.poserule --pack data/challenge.pack.tar --params configs/poserule.json \
    --sample sample_submission.csv --out submission_poserule.csv

# 2b. MViT-B + DETR: pack training and validation data, train, predict
python -m retailaction.pack --src <zip> --prefix train/train/ --ids train_ids.txt --out data/train.pack.tar
python -m retailaction.pack --src <zip> --prefix validation/validation/ --out data/validation.pack.tar
python -m retailaction.train --train data/train.pack.tar --val data/validation.pack.tar \
    --out runs/mvit --epochs 30 --bs 8 --accum 4
python -m retailaction.infer --ckpt runs/mvit/best.pt --pack data/challenge.pack.tar \
    --sample sample_submission.csv --out submission_mvit.csv

python -m retailaction.check_submission submission_mvit.csv --sample sample_submission.csv
```

Training takes a single 32 GB GPU (peak about 18 GB at batch 8).

To compare confidence scores and flip averaging on the validation split
without rerunning the model for each variant:

```bash
python -m retailaction.dump --ckpt runs/mvit/best.pt --pack data/validation.pack.tar --out val_raw.npz
python -m retailaction.post --raw val_raw.npz --pack data/validation.pack.tar
```

Point refinement (experimental): `--refine 2` adds two correction steps that
sample a 3x3 patch of stride-8 backbone features around each predicted point
at the query's temporal centre. The correction starts at zero, so a model
started with `--init runs/mvit/best.pt` gives the same output before
fine-tuning.

```bash
python -m retailaction.train --init runs/mvit/best.pt --refine 2 --train data/train.pack.tar \
    --val data/validation.pack.tar --out runs/refine --epochs 6 --lr 5e-5 --lr-backbone 1e-5 --warmup 100
```

The Dockerfile runs steps 1 and 2b inference end to end from the Kaggle zip
and a trained checkpoint mounted at `/ckpt/best.pt`.

## Licence

Code: MIT. The RetailAction data is under the Standard AI dataset licence and
the Kaggle competition rules; it is not redistributed here.
