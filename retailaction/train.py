"""Train the MViT-B + DETR baseline on packed RetailAction splits."""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .data import ClipDataset, collate
from .metric import evaluate, view_factors
from .model import RetailDETR, SetCriterion, to_actions


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = {}
    for x, _, ids in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.to(device, non_blocking=True))
        for sid, acts in zip(ids, to_actions(out)):
            preds[sid] = acts
    return preds


def validate(model, loader, device):
    ds = loader.dataset
    preds = predict(model, loader, device)
    gts, factors = {}, {}
    for pi, sid in ds.items:
        pack = ds.packs[pi]
        gts[sid] = pack.labels.get(sid) or []
        factors[sid] = view_factors(pack.metadata(sid))
    return evaluate(preds, gts, factors)["mAP"], preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"

    train_ds = ClipDataset(args.train, train=True)
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                          drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_dl = None
    if args.val:
        val_dl = DataLoader(ClipDataset(args.val), args.bs * 2, num_workers=args.workers, collate_fn=collate, pin_memory=True)
    print(f"train samples {len(train_ds)}, val samples {len(val_dl.dataset) if val_dl else 0}", flush=True)

    model = RetailDETR(num_queries=args.queries).to(device)
    crit = SetCriterion().to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": hd, "lr": args.lr}],
                            betas=(0.9, 0.999), weight_decay=args.wd)
    base_lrs = [args.lr_backbone, args.lr]
    total = args.epochs * (len(train_dl) // args.accum)
    step, best = 0, -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        run, n = {}, 0
        for it, (x, targets, _) in enumerate(train_dl):
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.to(device, non_blocking=True))
            loss, logs = crit(out, targets)
            (loss / args.accum).backward()
            for k, v in logs.items():
                run[k] = run.get(k, 0) + v
            n += 1
            if (it + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            f = min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / max(total, 1), 1.0)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                msg = " ".join(f"{k} {v / n:.4f}" for k, v in run.items())
                print(f"ep {epoch} step {step}/{total} {msg} lr {f * args.lr:.2e} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
                run, n = {}, 0
            if args.max_steps and step >= args.max_steps:
                break
        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if val_dl is not None and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs or args.max_steps):
            m, _ = validate(model, val_dl, device)
            print(f"epoch {epoch} val mAP {m:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            with open(os.path.join(args.out, "val_history.jsonl"), "a") as fh:
                fh.write(json.dumps({"epoch": epoch, "step": step, "val_mAP": m}) + "\n")
            if m > best:
                best = m
                torch.save(ckpt, os.path.join(args.out, "best.pt"))
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done, best val mAP {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
