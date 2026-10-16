"""Merge the raw dumps of several checkpoints into one dump.

Each dump is first folded with its own flip pass, then the queries of the
models are paired per sample with the Hungarian algorithm and averaged. The
output holds probabilities as logits (softmax of a log is the identity), so
post.py reads it with --tta none.

    python -m tools.ensemble_raw --raw a/test_raw.npz b/test_raw.npz --out ens_raw.npz
"""
import argparse

import numpy as np

from retailaction.post import merge_flip, softmax


def fold(raw):
    return merge_flip(softmax(raw["logits"]), raw["times"], raw["points"],
                      softmax(raw["logits_f"]), raw["times_f"], raw["points_f"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raws = [dict(np.load(p)) for p in args.raw]
    w = args.weights or [1.0] * len(raws)
    ids = raws[0]["ids"]
    assert all((r["ids"] == ids).all() for r in raws)
    prob, times, points = fold(raws[0])
    total = w[0]
    for r, wi in zip(raws[1:], w[1:]):
        p2, t2, x2 = fold(r)
        for i in range(prob.shape[0]):
            cost = (np.abs(times[i][:, None] - t2[i][None]).sum(-1)
                    + np.abs(points[i][:, None] - x2[i][None]).sum(-1)
                    + np.abs(prob[i][:, None, :-1] - p2[i][None, :, :-1]).sum(-1))
            from scipy.optimize import linear_sum_assignment
            a, b = linear_sum_assignment(cost)
            prob[i, a] = (total * prob[i, a] + wi * p2[i, b]) / (total + wi)
            times[i, a] = (total * times[i, a] + wi * t2[i, b]) / (total + wi)
            points[i, a] = (total * points[i, a] + wi * x2[i, b]) / (total + wi)
        total += wi
    np.savez(args.out, ids=ids, logits=np.log(np.clip(prob, 1e-9, None)), times=times, points=points)


if __name__ == "__main__":
    main()
