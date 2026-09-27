"""Local approximation of the challenge metric (mAP Global).

The official scorer is not public. This follows the published description:
a prediction matches a ground truth action when temporal IoU >= t, spatial
distance in meters <= d and score >= c; AP is computed from the precision and
recall over 101 confidence thresholds (VOC style monotone precision) and then
averaged over 10 temporal x 20 spatial thresholds. Poses here are in
normalized image coordinates, so the meters factor is meters per normalized
unit, averaged over bones and frames; the spatial distance is the mean of the
two views. Use it for relative comparisons only.
"""
import numpy as np

BONE_LENGTH_MEANS = {
    ("neck", "nose"): 0.19354,
    ("left_shoulder", "left_elbow"): 0.27096,
    ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210,
    ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204,
    ("left_knee", "left_ankle"): 0.39530,
    ("right_hip", "right_knee"): 0.39266,
    ("right_knee", "right_ankle"): 0.39322,
    ("left_shoulder", "right_shoulder"): 0.35484,
    ("left_hip", "right_hip"): 0.17150,
    ("neck", "left_shoulder"): 0.18136,
    ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_hip"): 0.51375,
    ("right_shoulder", "right_hip"): 0.51226,
}
T_THRESHOLDS = np.linspace(0.05, 0.5, 10)
D_THRESHOLDS = np.linspace(1.0, 0.05, 20)
C_THRESHOLDS = np.linspace(1.0, 0.0, 101)


def meters_per_unit(poses, min_score=0.3):
    """Meters per normalized image unit for one view, or None if no usable bones."""
    ratios = []
    for p in poses or []:
        if not p:
            continue
        joints, scores = p.get("pose") or {}, p.get("joint_scores") or {}
        for (a, b), m in BONE_LENGTH_MEANS.items():
            if a in joints and b in joints and scores.get(a, 0) >= min_score and scores.get(b, 0) >= min_score:
                l = float(np.hypot(joints[a][0] - joints[b][0], joints[a][1] - joints[b][1]))
                if l > 1e-4:
                    ratios.append(m / l)
    return float(np.median(ratios)) if ratios else None


def view_factors(metadata):
    cams = metadata.get("content", metadata).get("action_cam", {})
    return [meters_per_unit(cams.get(v, {}).get("poses")) for v in ("rank0", "rank1")]


def _pts(a):
    s = a["spatial"]["action_cam"]
    return np.array([s["rank0"]["x"], s["rank0"]["y"], s["rank1"]["x"], s["rank1"]["y"]], dtype=np.float64)


def evaluate(preds, gts, factors):
    """preds, gts: {id: [action dicts]}. factors: {id: [f_rank0, f_rank1]} meters per unit (None allowed)."""
    known = [f for fs in factors.values() for f in fs if f]
    fallback = float(np.median(known)) if known else 1.0
    n_gt = sum(len(g) for g in gts.values())
    # Pairwise tIoU and distance for every prediction against the gts of the same video.
    records = []  # (score, video index, tiou vector, dist vector)
    for vid, g in gts.items():
        f = [x or fallback for x in factors.get(vid, [None, None])]
        gt_t = np.array([[a["start"], a["end"]] for a in g]).reshape(-1, 2)
        gt_p = np.array([_pts(a) for a in g]).reshape(-1, 4)
        for a in preds.get(vid, []):
            s, e = a["start"], a["end"]
            inter = np.clip(np.minimum(e, gt_t[:, 1]) - np.maximum(s, gt_t[:, 0]), 0, None)
            union = (e - s) + (gt_t[:, 1] - gt_t[:, 0]) - inter
            tiou = inter / np.maximum(union, 1e-9)
            p = _pts(a)
            d0 = np.hypot(*(p[:2] - gt_p[:, :2]).T) * f[0]
            d1 = np.hypot(*(p[2:] - gt_p[:, 2:]).T) * f[1]
            records.append((a["score"], vid, tiou, (d0 + d1) / 2))
    records.sort(key=lambda r: -r[0])
    scores = np.array([r[0] for r in records])
    aps = np.zeros((len(T_THRESHOLDS), len(D_THRESHOLDS)))
    for i, t in enumerate(T_THRESHOLDS):
        for j, d in enumerate(D_THRESHOLDS):
            used = {vid: np.zeros(len(g), bool) for vid, g in gts.items()}
            tp = np.zeros(len(records))
            for k, (_, vid, tiou, dist) in enumerate(records):
                ok = (tiou >= t) & (dist <= d) & ~used[vid]
                if ok.any():
                    cand = np.where(ok)[0]
                    best = cand[np.argmax(tiou[cand])]
                    used[vid][best] = True
                    tp[k] = 1
            ctp = np.cumsum(tp)
            prec, rec = [], []
            for c in C_THRESHOLDS:
                n = int((scores >= c).sum())
                tpc = ctp[n - 1] if n > 0 else 0.0
                prec.append(tpc / n if n > 0 else 1.0)
                rec.append(tpc / max(n_gt, 1))
            prec, rec = np.array(prec), np.array(rec)
            # monotone precision envelope, integrate over recall
            for k in range(len(prec) - 2, -1, -1):
                prec[k] = max(prec[k], prec[k + 1])
            r_prev, ap = 0.0, 0.0
            for p_, r_ in zip(prec, rec):
                ap += max(r_ - r_prev, 0) * p_
                r_prev = max(r_prev, r_)
            aps[i, j] = ap
    return {"mAP": float(aps.mean()), "ap_matrix": aps}
