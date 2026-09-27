"""Dataset over packed tars written by retailaction.pack."""
import gzip
import io
import json
import os
import random

import av
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ("take", "put", "touch")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


class Pack:
    """Random access to one packed tar through its index file."""

    def __init__(self, path):
        self.path = path
        with open(path + ".index.json") as f:
            d = json.load(f)
        self.samples = d["samples"]
        self.labels = d.get("labels", {})
        self._fh, self._pid = None, None

    def read(self, sid, name):
        # one handle per process: forked dataloader workers must not share a file offset
        if self._fh is None or self._pid != os.getpid():
            self._fh, self._pid = open(self.path, "rb"), os.getpid()
        off, size = self.samples[sid][name]
        self._fh.seek(off)
        return self._fh.read(size)

    def metadata(self, sid):
        return json.loads(gzip.decompress(self.read(sid, "metadata.json.gz")))

    def __getstate__(self):
        s = dict(self.__dict__)
        s["_fh"], s["_pid"] = None, None
        return s


def decode(data):
    with av.open(io.BytesIO(data)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def sample_indices(n, t, offset):
    return np.minimum(((np.arange(t) + offset) * n / t).astype(int), n - 1)


def points_of(labels, view):
    return [(a["spatial"]["action_cam"][view]["x"], a["spatial"]["action_cam"][view]["y"]) for a in labels]


class ClipDataset(Dataset):
    def __init__(self, pack_paths, train=False, num_frames=16, size=224, ids=None):
        self.packs = [Pack(p) for p in pack_paths]
        self.items = []
        for pi, p in enumerate(self.packs):
            for sid in sorted(p.samples):
                if ids is None or sid in ids:
                    self.items.append((pi, sid))
        self.train, self.t, self.size = train, num_frames, size

    def __len__(self):
        return len(self.items)

    def _view(self, frames, pts):
        """frames: (N, H, W, 3) uint8. pts: normalized (x, y) targets in this view."""
        n, h, w, _ = frames.shape
        offset = random.random() if self.train else 0.5
        x = torch.from_numpy(frames[sample_indices(n, self.t, offset)]).permute(3, 0, 1, 2).float() / 255.0
        x0, y0, s = 0.0, 0.0, 1.0
        if self.train:
            for _ in range(10):
                cs = random.uniform(0.8, 1.0)
                cx, cy = random.uniform(0, 1 - cs), random.uniform(0, 1 - cs)
                if all(cx <= px <= cx + cs and cy <= py <= cy + cs for px, py in pts):
                    x0, y0, s = cx, cy, cs
                    break
            if s < 1.0:
                x = x[:, :, int(y0 * h):int((y0 + s) * h), int(x0 * w):int((x0 + s) * w)]
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        flip = self.train and random.random() < 0.5
        if flip:
            x = x.flip(-1)
        new_pts = []
        for px, py in pts:
            px, py = (px - x0) / s, (py - y0) / s
            new_pts.append((1.0 - px if flip else px, py))
        x = (x - MEAN) / STD
        return x, new_pts

    def __getitem__(self, i):
        pi, sid = self.items[i]
        pack = self.packs[pi]
        labels = pack.labels.get(sid) or []
        views, pts = [], []
        for v, name in (("rank0", "r0.mp4"), ("rank1", "r1.mp4")):
            x, p = self._view(decode(pack.read(sid, name)), points_of(labels, v))
            views.append(x)
            pts.append(p)
        target = {
            "labels": torch.tensor([CLASSES.index(a["label"]) if a.get("label") in CLASSES else 0 for a in labels], dtype=torch.long),
            "times": torch.tensor([[a["start"], a["end"]] for a in labels], dtype=torch.float32).view(-1, 2),
            # (num_actions, 4): rank0 x, rank0 y, rank1 x, rank1 y
            "points": torch.tensor([list(pts[0][k]) + list(pts[1][k]) for k in range(len(labels))], dtype=torch.float32).view(-1, 4),
        }
        return torch.stack(views), target, sid


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]
