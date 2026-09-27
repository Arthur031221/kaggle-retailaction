"""Run a trained checkpoint on a packed split and write a submission CSV."""
import argparse
import csv
import json

import torch
from torch.utils.data import DataLoader

from .data import ClipDataset, collate
from .model import RetailDETR
from .train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", nargs="+", required=True)
    ap.add_argument("--sample", required=True, help="sample_submission.csv, used for id order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = RetailDETR(num_queries=ck["args"].get("queries", 5), pretrained=False)
    model.load_state_dict(ck["model"])
    model.cuda()
    dl = DataLoader(ClipDataset(args.pack), args.bs, num_workers=args.workers, collate_fn=collate)
    preds = predict(model, dl, "cuda")

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
