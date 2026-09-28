"""MViT-B backbone with a DETR style multi-view head, following the RetailAction paper.

Both views go through a shared MViT-B (Kinetics-400 weights from torchvision).
Tokens are projected, given learned 3D positional encodings and refined by one
transformer encoder per view. The two encoded token sets are concatenated with
a learned view embedding and read by a DETR decoder with a few action queries.
Each query predicts class logits (take, put, touch, no action), a temporal
interval and one (x, y) point per view. With refine > 0, each point is corrected by
sampling a 3x3 patch of stride-8 backbone features around it (PointRefiner).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

NUM_CLASSES = 3


class MViTTokens(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = mvit_v1_b(weights=MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None)
        self.conv_proj, self.pos_encoding, self.blocks, self.norm = m.conv_proj, m.pos_encoding, m.blocks, m.norm
        self.dim = m.norm.normalized_shape[0]

        self.fine_dim = m.blocks[3].norm1.normalized_shape[0]  # input of the block that pools to stride 16

    def forward(self, x):
        """Returns the final tokens without the class token, their (T, H, W), and the last
        stride-8 feature map (N, C, T, H, W) taken before the block that pools to stride 16."""
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        fine = None
        base = thw[1]
        for block in self.blocks:
            x, thw = block(x, thw)
            if thw[1] == base // 2:
                fine = (x, thw)
        x = self.norm(x)
        f, (t, h, w) = fine
        fine = f[:, 1:].transpose(1, 2).reshape(f.shape[0], f.shape[2], t, h, w)
        return x[:, 1:], thw, fine


class MLP(nn.Module):
    def __init__(self, d, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out))

    def forward(self, x):
        return self.net(x)


class PointRefiner(nn.Module):
    """One refinement step: sample a 3x3 patch of stride-8 features around each view's point
    at the query's temporal centre, and predict a correction in logit space."""

    def __init__(self, d, c, patch=3):
        super().__init__()
        r = torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2
        self.register_buffer("offsets", torch.stack(torch.meshgrid(r, r, indexing="xy"), -1).view(-1, 2), persistent=False)
        self.patch_proj = nn.Linear(patch * patch * c, d)
        self.delta = MLP(3 * d, 4)
        nn.init.zeros_(self.delta.net[-1].weight)
        nn.init.zeros_(self.delta.net[-1].bias)

    def sample(self, feat, pts, tc):
        """feat: (B, C, T, H, W), pts: (B, Q, 2) in [0, 1], tc: (B, Q) -> (B, Q, 9 C)."""
        b, c, _, h, w = feat.shape
        q, k = pts.shape[1], self.offsets.shape[0]
        step = 2.0 / torch.tensor([w, h], device=pts.device, dtype=pts.dtype)
        xy = (pts * 2 - 1)[:, :, None] + self.offsets.to(pts.dtype) * step  # (B, Q, K, 2)
        t = (tc * 2 - 1)[:, :, None, None].expand(-1, -1, k, 1)
        grid = torch.cat([xy, t], -1).view(b, 1, q, k, 3)
        s = F.grid_sample(feat, grid.to(feat.dtype), align_corners=False)  # (B, C, 1, Q, K)
        return s[:, :, 0].permute(0, 2, 3, 1).reshape(b, q, k * c)

    def forward(self, h, points, times, feats):
        """feats: (B, 2, C, T, H, W). Returns refined points (B, Q, 4)."""
        p = points.detach()
        tc = times.detach().mean(-1)
        f0 = self.patch_proj(self.sample(feats[:, 0], p[..., :2], tc))
        f1 = self.patch_proj(self.sample(feats[:, 1], p[..., 2:], tc))
        d = self.delta(torch.cat([h, f0, f1], -1))
        return (torch.logit(p.float().clamp(1e-4, 1 - 1e-4)) + d.float()).sigmoid()


class RetailDETR(nn.Module):
    def __init__(self, d=256, num_queries=5, enc_layers=2, dec_layers=6, heads=8, pretrained=True, grid=(8, 7, 7),
                 refine=0):
        super().__init__()
        self.backbone = MViTTokens(pretrained)
        self.refiners = nn.ModuleList([PointRefiner(d, self.backbone.fine_dim) for _ in range(refine)])
        self.proj = nn.Linear(self.backbone.dim, d)
        t, h, w = grid
        self.pos_t = nn.Parameter(torch.randn(t, d) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(h, d) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(w, d) * 0.02)
        self.view_emb = nn.Parameter(torch.randn(2, d) * 0.02)
        mk_enc = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), enc_layers)
        self.encoders = nn.ModuleList([mk_enc(), mk_enc()])
        self.dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True) for _ in range(dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)
        self.cls_head = nn.Linear(d, NUM_CLASSES + 1)
        self.point_head = MLP(d, 4)
        self.time_head = MLP(d, 2)

    def heads(self, h):
        tc = self.time_head(h).sigmoid()
        center, width = tc[..., 0], tc[..., 1]
        times = torch.stack([(center - width / 2).clamp(0, 1), (center + width / 2).clamp(0, 1)], -1)
        return {"logits": self.cls_head(h), "points": self.point_head(h).sigmoid(), "times": times}

    def forward(self, x):
        """x: (B, 2 views, 3, T, H, W)."""
        b = x.shape[0]
        tok, thw, fine = self.backbone(x.flatten(0, 1))
        tok = self.proj(tok)
        pos = (self.pos_t[:, None, None] + self.pos_h[None, :, None] + self.pos_w[None, None, :]).flatten(0, 2)
        tok = (tok + pos[: tok.shape[1]]).view(b, 2, tok.shape[1], -1)
        mem = torch.cat([self.encoders[v](tok[:, v]) + self.view_emb[v] for v in range(2)], 1)
        h = self.queries.unsqueeze(0).expand(b, -1, -1)
        outs = []
        for layer in self.dec_layers:
            h = layer(h, mem)
            outs.append(self.heads(self.dec_norm(h)))
        if self.refiners:
            hn = self.dec_norm(h)
            feats = fine.view(b, 2, *fine.shape[1:])
            last = outs[-1]
            for ref in self.refiners:
                last = dict(last, points=ref(hn, last["points"], last["times"], feats))
                outs.append(last)
        out = outs[-1]
        out["aux"] = outs[:-1]
        return out


def temporal_iou(a, b):
    """a: (N, 2), b: (M, 2) -> (N, M)."""
    inter = (torch.min(a[:, None, 1], b[None, :, 1]) - torch.max(a[:, None, 0], b[None, :, 0])).clamp(min=0)
    union = (a[:, None, 1] - a[:, None, 0]) + (b[None, :, 1] - b[None, :, 0]) - inter
    return inter / union.clamp(min=1e-6)


def point_dist(p, g):
    """Sum over the two views of the L2 distance. p: (N, 4), g: (M, 4) -> (N, M)."""
    d0 = torch.cdist(p[:, :2], g[:, :2])
    d1 = torch.cdist(p[:, 2:], g[:, 2:])
    return d0 + d1


class SetCriterion(nn.Module):
    def __init__(self, w_cls=1.0, w_point=5.0, w_l1=5.0, w_iou=2.0, eos_coef=0.1):
        super().__init__()
        self.w_cls, self.w_point, self.w_l1, self.w_iou = w_cls, w_point, w_l1, w_iou
        weight = torch.ones(NUM_CLASSES + 1)
        weight[-1] = eos_coef
        self.register_buffer("cls_weight", weight)

    @torch.no_grad()
    def match(self, out, targets):
        idx = []
        prob = out["logits"].float().softmax(-1)
        for i, t in enumerate(targets):
            if len(t["labels"]) == 0:
                idx.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            c = (-self.w_cls * prob[i][:, t["labels"]]
                 + self.w_point * point_dist(out["points"][i].float(), t["points"])
                 + self.w_l1 * torch.cdist(out["times"][i].float(), t["times"], p=1)
                 - self.w_iou * temporal_iou(out["times"][i].float(), t["times"]))
            r, cidx = linear_sum_assignment(c.cpu().numpy())
            idx.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(cidx, dtype=torch.long)))
        return idx

    def single(self, out, targets):
        idx = self.match(out, targets)
        logits = out["logits"].float()
        tgt_cls = torch.full(logits.shape[:2], NUM_CLASSES, dtype=torch.long, device=logits.device)
        pp, gp, pt, gt = [], [], [], []
        for i, (r, c) in enumerate(idx):
            if len(r) == 0:
                continue
            tgt_cls[i, r] = targets[i]["labels"][c]
            pp.append(out["points"][i, r].float())
            gp.append(targets[i]["points"][c])
            pt.append(out["times"][i, r].float())
            gt.append(targets[i]["times"][c])
        loss_cls = F.cross_entropy(logits.transpose(1, 2), tgt_cls, weight=self.cls_weight)
        if not pp:
            zero = logits.sum() * 0
            return {"cls": loss_cls, "point": zero, "l1": zero, "iou": zero}
        pp, gp, pt, gt = map(torch.cat, (pp, gp, pt, gt))
        n = max(len(pp), 1)
        loss_point = ((pp[:, :2] - gp[:, :2]).norm(dim=-1) + (pp[:, 2:] - gp[:, 2:]).norm(dim=-1)).sum() / n
        loss_l1 = (pt - gt).abs().sum() / n
        loss_iou = (1 - temporal_iou(pt, gt).diagonal()).sum() / n
        return {"cls": loss_cls, "point": loss_point, "l1": loss_l1, "iou": loss_iou}

    def forward(self, out, targets):
        total, logs = 0.0, {}
        for k, o in enumerate([out] + out.get("aux", [])):
            ls = self.single(o, targets)
            total = total + self.w_cls * ls["cls"] + self.w_point * ls["point"] + self.w_l1 * ls["l1"] + self.w_iou * ls["iou"]
            if k == 0:
                logs = {k2: v.item() for k2, v in ls.items()}
        return total, logs


def to_actions(out, max_actions=None):
    """Convert model outputs to per-sample lists of submission dicts."""
    prob = out["logits"].float().softmax(-1)
    score = prob[..., :NUM_CLASSES].max(-1).values
    res = []
    for i in range(prob.shape[0]):
        acts = []
        order = score[i].argsort(descending=True)
        for q in order.tolist():
            s, e = out["times"][i, q].tolist()
            if e - s < 1e-3:
                c = (s + e) / 2
                s, e = max(0.0, c - 5e-4), min(1.0, c + 5e-4)
            p = out["points"][i, q].tolist()
            acts.append({
                "score": float(score[i, q]), "start": float(s), "end": float(e),
                "spatial": {"action_cam": {"rank0": {"x": p[0], "y": p[1]}, "rank1": {"x": p[2], "y": p[3]}}},
            })
        res.append(acts[:max_actions] if max_actions else acts)
    return res
