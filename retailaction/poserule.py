"""Pose rule: predict the interaction from the shopper's 2D hand keypoints.

No learning and no video decoding. For every frame and each hand we measure arm
extension (hand to neck distance divided by torso length) in both views. The
frame and hand with the largest extension, summed over the two views, is taken
as the contact moment. The touch point in each view is that hand's keypoint
and the interval is a fixed width window around the contact frame. Weaker
secondary peaks are added with lower scores.

Pose keypoints in the metadata are stored as [row, col] in normalized image
coordinates, while labels use (x, y) = (col, row); this module swaps them.
"""
import numpy as np

HANDS = (("left_hand", "left_wrist"), ("right_hand", "right_wrist"))

DEFAULT = {
    "width": 0.22,        # interval width (fraction of the clip)
    "shift": 0.0,         # interval center = (t* + 0.5) / n + shift
    "smooth": 1,          # moving average half window, in frames
    "min_gap": 0.25,      # secondary peaks must be this far (fraction of clip) from earlier ones
    "max_actions": 3,
    "second_scale": 0.5,  # score multiplier for secondary peaks
    "other_hand_scale": 0.3,  # score multiplier for the other hand at the main peak (0 disables)
    "score_center": 1.2,  # score = sigmoid((ext - center) * slope)
    "score_slope": 3.0,
    "extra_widths": (),   # extra copies of the main prediction with other widths, lower scores
    "reach": 0.0,         # move the point along neck->hand by this fraction of that vector
    "adapt_thr": 0.0,     # >0: interval from frames whose extension >= adapt_thr * peak
    "adapt_mix": 0.5,     # blend between adaptive and fixed width intervals
}


def _xy(p, name):
    v = p.get(name)
    return None if v is None else (float(v[1]), float(v[0]))


def view_arrays(poses, n):
    """Return hands (n, 2, 2), neck (n, 2), torso length (scalar) for one view."""
    hands = np.full((n, 2, 2), np.nan)
    neck = np.full((n, 2), np.nan)
    torso = []
    for t in range(min(n, len(poses or []))):
        p = ((poses[t] or {}).get("pose")) or {}
        for h, names in enumerate(HANDS):
            for name in names:
                q = _xy(p, name)
                if q is not None:
                    hands[t, h] = q
                    break
        q = _xy(p, "neck")
        if q is not None:
            neck[t] = q
        w = _xy(p, "middle_of_waist")
        if q is not None and w is not None:
            torso.append(np.hypot(q[0] - w[0], q[1] - w[1]))
        ls, rs = _xy(p, "left_shoulder"), _xy(p, "right_shoulder")
        if ls is not None and rs is not None:
            torso.append(1.4 * np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
    scale = float(np.median(torso)) if torso else 0.15
    return hands, neck, max(scale, 0.03)


def _fill(a):
    """Fill NaN rows by nearest valid row along axis 0."""
    a = a.copy()
    valid = ~np.isnan(a).any(-1)
    if not valid.any():
        return a
    idx = np.where(valid)[0]
    for t in np.where(~valid)[0]:
        a[t] = a[idx[np.argmin(np.abs(idx - t))]]
    return a


def predict_one(metadata, prm=DEFAULT):
    cams = metadata.get("content", metadata).get("action_cam", {})
    views = []
    for v in ("rank0", "rank1"):
        c = cams.get(v, {})
        poses = c.get("poses") or []
        n = max(len(c.get("frame_timestamps") or []), len(poses), 1)
        views.append((n,) + view_arrays(poses, n))
    n = max(v[0] for v in views)
    ext = np.full((2, n, 2), np.nan)  # view, frame, hand
    for vi, (nv, hands, neck, scale) in enumerate(views):
        neck_f = _fill(neck)
        d = np.linalg.norm(hands - neck_f[:, None, :], axis=-1) / scale
        # resample to the common frame count if the views differ
        src = (np.arange(nv) + 0.5) / nv
        dst = (np.arange(n) + 0.5) / n
        for h in range(2):
            col = d[:, h]
            ok = ~np.isnan(col)
            if ok.sum() >= 1:
                ext[vi, :, h] = np.interp(dst, src[ok], col[ok])
    if np.isnan(ext).all():
        return []
    sig = np.nanmean(ext, axis=0)  # frame, hand (mean over views that have data)
    if np.isnan(sig).all():
        return []
    sig = np.nan_to_num(sig, nan=0.0)
    k = int(prm["smooth"])
    if k > 0:
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        sig = np.stack([np.convolve(np.pad(sig[:, h], k, mode="edge"), ker, mode="valid") for h in range(2)], 1)

    def points(t, h):
        out = []
        for vi, (nv, hands, neck, _) in enumerate(views):
            tv = min(int((t + 0.5) / n * nv), nv - 1)
            hv = _fill(hands[:, h])
            if np.isnan(hv[tv]).any():
                alt = _fill(hands[:, 1 - h])
                q = alt[tv] if not np.isnan(alt[tv]).any() else np.array([0.5, 0.5])
            else:
                q = hv[tv]
            nk = _fill(neck)[tv]
            if prm["reach"] and not np.isnan(nk).any():
                q = q + prm["reach"] * (q - nk)
            out.append((float(np.clip(q[0], 0, 1)), float(np.clip(q[1], 0, 1))))
        return out

    def action(t, h, score, width):
        c = (t + 0.5) / n + prm["shift"]
        s, e = c - width / 2, c + width / 2
        if prm["adapt_thr"] > 0:
            col = sig[:, h]
            thr = prm["adapt_thr"] * col[t]
            a = t
            while a > 0 and col[a - 1] >= thr:
                a -= 1
            b = t
            while b < n - 1 and col[b + 1] >= thr:
                b += 1
            m = prm["adapt_mix"]
            s = m * (a / n) + (1 - m) * s
            e = m * ((b + 1) / n) + (1 - m) * e
        s, e = max(0.0, s), min(1.0, e)
        if e - s < 1e-3:
            s, e = max(0.0, e - 1e-3), min(1.0, s + 1e-3)
        p = points(t, h)
        return {"score": float(score), "start": s, "end": e,
                "spatial": {"action_cam": {"rank0": {"x": p[0][0], "y": p[0][1]}, "rank1": {"x": p[1][0], "y": p[1][1]}}}}

    def to_score(x):
        return 1.0 / (1.0 + np.exp(-(x - prm["score_center"]) * prm["score_slope"]))

    best = sig.max(1)
    order = np.argsort(-best)
    peaks = []
    for t in order:
        if len(peaks) >= prm["max_actions"]:
            break
        if all(abs(t - q) / n >= prm["min_gap"] for q in peaks):
            peaks.append(int(t))
    acts = []
    for i, t in enumerate(peaks):
        h = int(np.argmax(sig[t]))
        sc = to_score(sig[t, h]) * (1.0 if i == 0 else prm["second_scale"])
        acts.append(action(t, h, sc, prm["width"]))
        if i == 0:
            if prm["other_hand_scale"] > 0:
                acts.append(action(t, 1 - h, to_score(sig[t, 1 - h]) * prm["other_hand_scale"], prm["width"]))
            for j, w in enumerate(prm["extra_widths"]):
                acts.append(action(t, h, sc * 0.2 / (j + 1), w))
    return acts


def main():
    import argparse
    import csv
    import json
    import warnings

    from .data import Pack

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description="Write a submission with the pose rule")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--params", default="")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prm = dict(DEFAULT)
    if args.params:
        with open(args.params) as f:
            prm.update(json.load(f))
    pack = Pack(args.pack)
    with open(args.sample, newline="") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        for sid in ids:
            acts = predict_one(pack.metadata(sid), prm) if sid in pack.samples else []
            w.writerow([sid, json.dumps({"actions": acts})])
    print(f"wrote {len(ids)} rows to {args.out}")


if __name__ == "__main__":
    main()
