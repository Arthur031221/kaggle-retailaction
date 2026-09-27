"""Check whether challenge test clips also appear in a labelled split.

Only rank0 videos are read (no metadata, so no labels). Each clip becomes a
small grayscale thumbnail of its first, middle and last frames; we then look at
nearest neighbour distances between the challenge clips and the other split.
Re-encoded copies of the same clip give near zero distances.

Example:
  python tools/overlap_check.py --challenge-pack challenge.pack.tar \
      --zip @/path/url.txt --prefix test/ --out overlap_test.json
"""
import argparse
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retailaction.data import Pack  # noqa: E402
from retailaction.remote import HttpRangeFile  # noqa: E402


def thumb(data, s=16):
    with av.open(io.BytesIO(data)) as c:
        frames = [f.to_ndarray(format="gray", width=s, height=s).astype(np.float32) for f in c.decode(video=0)]
    v = np.concatenate([frames[0].ravel(), frames[len(frames) // 2].ravel(), frames[-1].ravel(), [len(frames) * 4.0]])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge-pack", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ch = Pack(args.challenge_pack)
    ch_ids = sorted(ch.samples)
    raw = [ch.read(s, "r0.mp4") for s in ch_ids]  # sequential: the pack handle is not thread safe
    with ThreadPoolExecutor(8) as ex:
        A = np.stack(list(ex.map(thumb, raw)))

    src = open(args.zip[1:]).read().strip() if args.zip.startswith("@") else args.zip
    zf = zipfile.ZipFile(HttpRangeFile(src) if src.startswith("http") else src)
    infos = sorted([i for i in zf.infolist() if i.filename.startswith(args.prefix) and i.filename.endswith("rank0_video.mp4")],
                   key=lambda i: i.header_offset)
    names = [i.filename for i in infos]
    datas = (zf.read(i) for i in infos)
    with ThreadPoolExecutor(8) as ex:
        B = np.stack(list(ex.map(thumb, datas)))

    np.savez_compressed(args.out + ".thumbs.npz", A=A, B=B, ch_ids=np.array(ch_ids), names=np.array(names))

    def nn(X, Y, same=False):
        # RMS distance through the Gram matrix; avoids a (len(X), len(Y), dim) array
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        sq = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
        d = np.sqrt(np.clip(sq, 0, None) / X.shape[1])
        if same:
            np.fill_diagonal(d, np.inf)
        return d.min(1), d.argmin(1)

    d_cross, j = nn(A, B)
    d_self, _ = nn(A, A, same=True)
    res = {
        "n_challenge": len(A), "n_other": len(B), "prefix": args.prefix,
        "cross_nn_percentiles": np.percentile(d_cross, [0, 1, 5, 25, 50]).round(2).tolist(),
        "challenge_self_nn_percentiles": np.percentile(d_self, [0, 1, 5, 25, 50]).round(2).tolist(),
        "cross_below_3": int((d_cross < 3).sum()),
        "closest": [[ch_ids[i], names[j[i]], round(float(d_cross[i]), 2)] for i in np.argsort(d_cross)[:10]],
    }
    print(json.dumps(res, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
