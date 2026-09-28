"""Save a checkpoint's raw query outputs on a packed split, with and without a
horizontal flip, so that scoring and test time augmentation can be compared on
the CPU without running the model again.

Output npz: ids, and for each pass (plain, flip) logits (N, Q, C+1),
times (N, Q, 2) and points (N, Q, 4). Flipped points are mapped back to the
original frame (x -> 1 - x in both views).
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClipDataset, collate
from .model import RetailDETR


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = RetailDETR(num_queries=ck["args"].get("queries", 5), refine=ck["args"].get("refine", 0), pretrained=False)
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    dl = DataLoader(ClipDataset(args.pack), args.bs, num_workers=args.workers, collate_fn=collate)

    ids, res = [], {k: [] for k in ("logits", "times", "points", "logits_f", "times_f", "points_f")}
    for x, _, sids in dl:
        x = x.cuda(non_blocking=True)
        for suffix, inp in (("", x), ("_f", x.flip(-1))):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inp)
            pts = out["points"].float()
            if suffix:
                pts = pts.clone()
                pts[..., 0::2] = 1.0 - pts[..., 0::2]
            res["logits" + suffix].append(out["logits"].float().cpu().numpy())
            res["times" + suffix].append(out["times"].float().cpu().numpy())
            res["points" + suffix].append(pts.cpu().numpy())
        ids.extend(sids)
    np.savez_compressed(args.out, ids=np.array(ids), **{k: np.concatenate(v) for k, v in res.items()})
    print(f"saved {len(ids)} samples to {args.out}")


if __name__ == "__main__":
    main()
