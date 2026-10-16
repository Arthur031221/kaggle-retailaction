#!/bin/sh
# Test-set inference for the final submission: two checkpoints, each run with
# the plain and flipped pass, folded and averaged, then foreground rescoring.
# Expects the Kaggle download in $DATA, base.pt and refined.pt in $CKPT and
# writes to $OUT. Pass PACK=<file> to reuse an already packed test set.
set -e
DATA=${DATA:-/data}
CKPT=${CKPT:-/ckpt}
OUT=${OUT:-/out}
PACK=${PACK:-$OUT/challenge.pack.tar}
[ -f "$PACK" ] || python -m retailaction.pack --src $DATA/retail-action-challenge-praw-workshop-wacv-2027.zip \
    --prefix challenge_test_set/ --out $PACK
python -m retailaction.dump --ckpt $CKPT/base.pt --pack $PACK --out $OUT/base.npz
python -m retailaction.dump --ckpt $CKPT/refined.pt --pack $PACK --out $OUT/refined.npz
python -m tools.ensemble_raw --raw $OUT/base.npz $OUT/refined.npz --out $OUT/ensemble.npz
python - $OUT/ensemble.npz <<'PY'
import sys
import numpy as np
z = dict(np.load(sys.argv[1]))
z["logits_f"], z["times_f"], z["points_f"] = z["logits"], z["times"], z["points"]
np.savez(sys.argv[1], **z)
PY
python -m retailaction.post --raw $OUT/ensemble.npz --score fg --tta none \
    --sample $DATA/sample_submission.csv --out $OUT/submission.csv
python -m retailaction.check_submission $OUT/submission.csv --sample $DATA/sample_submission.csv
