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
ZIP=$DATA/retail-action-challenge-praw-workshop-wacv-2027.zip
SAMPLE=${SAMPLE:-$DATA/sample_submission.csv}
if [ ! -f "$SAMPLE" ]; then
    # The Kaggle download keeps the sample submission inside the zip.
    python - "$ZIP" "$OUT/sample_submission.csv" <<'PY'
import sys
import zipfile
z = zipfile.ZipFile(sys.argv[1])
name = next(n for n in z.namelist() if n.endswith("sample_submission.csv"))
open(sys.argv[2], "wb").write(z.read(name))
PY
    SAMPLE=$OUT/sample_submission.csv
fi
# The index is written last, so a pack cut short by an earlier failure is redone.
[ -f "$PACK.index.json" ] || python -m retailaction.pack --src $ZIP \
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
    --sample $SAMPLE --out $OUT/submission.csv
python -m retailaction.check_submission $OUT/submission.csv --sample $SAMPLE
