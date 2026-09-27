"""Turn raw query outputs from dump.py into actions: choose the confidence
score, optionally merge the flipped pass, then either evaluate on a labelled
pack or write a submission CSV.

    python -m retailaction.post --raw val_raw.npz --pack validation.pack.tar
    python -m retailaction.post --raw test_raw.npz --score fg --tta flip \
        --sample sample_submission.csv --out submission.csv
"""
import argparse
import csv
import itertools
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metric import D_THRESHOLDS, T_THRESHOLDS, evaluate, view_factors


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def merge_flip(prob, times, points, prob_f, times_f, points_f):
    """Pair the queries of the two passes per sample and average them."""
    prob, times, points = prob.copy(), times.copy(), points.copy()
    for i in range(prob.shape[0]):
        cost = (np.abs(times[i][:, None] - times_f[i][None]).sum(-1)
                + np.abs(points[i][:, None] - points_f[i][None]).sum(-1)
                + np.abs(prob[i][:, None, :-1] - prob_f[i][None, :, :-1]).sum(-1))
        r, c = linear_sum_assignment(cost)
        prob[i, r] = (prob[i, r] + prob_f[i, c]) / 2
        times[i, r] = (times[i, r] + times_f[i, c]) / 2
        points[i, r] = (points[i, r] + points_f[i, c]) / 2
    return prob, times, points


def actions(raw, score="max", tta="none"):
    prob, times, points = softmax(raw["logits"]), raw["times"], raw["points"]
    if tta == "flip":
        prob, times, points = merge_flip(prob, times, points, softmax(raw["logits_f"]), raw["times_f"], raw["points_f"])
    conf = prob[..., :-1].max(-1) if score == "max" else 1.0 - prob[..., -1]
    preds = {}
    for i, sid in enumerate(raw["ids"].tolist()):
        acts = []
        for q in np.argsort(-conf[i]):
            s, e = float(times[i, q, 0]), float(times[i, q, 1])
            if e - s < 1e-3:
                c = (s + e) / 2
                s, e = max(0.0, c - 5e-4), min(1.0, c + 5e-4)
            p = points[i, q].tolist()
            acts.append({
                "score": float(conf[i, q]), "start": s, "end": e,
                "spatial": {"action_cam": {"rank0": {"x": p[0], "y": p[1]}, "rank1": {"x": p[2], "y": p[3]}}},
            })
        preds[sid] = acts
    return preds


def breakdown(ap_matrix):
    """Coarse view of where AP is lost: loose corner, point axis, time axis."""
    t, d = np.asarray(T_THRESHOLDS), np.asarray(D_THRESHOLDS)
    loose_t, loose_d = int(np.argmin(t)), int(np.argmax(d))
    return {
        "loose": float(ap_matrix[loose_t, loose_d]),
        "points_at_loose_t": float(ap_matrix[loose_t].mean()),
        "time_at_loose_d": float(ap_matrix[:, loose_d].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pack", help="labelled pack: evaluate every score x tta combination")
    ap.add_argument("--score", choices=["max", "fg"], default="max")
    ap.add_argument("--tta", choices=["none", "flip"], default="none")
    ap.add_argument("--sample")
    ap.add_argument("--out")
    args = ap.parse_args()
    raw = dict(np.load(args.raw))

    if args.pack:
        from .data import Pack
        pack = Pack(args.pack)
        ids = raw["ids"].tolist()
        gts = {sid: pack.labels.get(sid) or [] for sid in ids}
        factors = {sid: view_factors(pack.metadata(sid)) for sid in ids}
        for score, tta in itertools.product(("max", "fg"), ("none", "flip")):
            preds = actions(raw, score, tta)
            for rule in ("min", "mean"):
                r = evaluate(preds, gts, factors, view_rule=rule)
                print(json.dumps({"score": score, "tta": tta, "view_rule": rule, "mAP": round(r["mAP"], 4),
                                  **{k: round(v, 4) for k, v in breakdown(r["ap_matrix"]).items()}}))
        return

    preds = actions(raw, args.score, args.tta)
    with open(args.sample, newline="") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    missing = [i for i in ids if i not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} ids have no prediction, for example {missing[:5]}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        for i in ids:
            w.writerow([i, json.dumps({"actions": preds[i]})])
    print(f"wrote {len(ids)} rows to {args.out}")


if __name__ == "__main__":
    main()
