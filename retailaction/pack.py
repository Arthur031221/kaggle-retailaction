"""Pack RetailAction samples into one compact tar plus a JSON index.

Each sample directory holds rank0_video.mp4, rank1_video.mp4 and metadata.json
(the Kaggle challenge test set also has per-view *_poses.json files). Videos are
re-encoded to 256x256 H.264 with every frame kept, and JSON files are gzipped.
The output tar is read by offset at training and inference time, so nothing
needs to be extracted.

Sources can be an HF split tar (path, or "-" for a stream on stdin), a zip
(the Kaggle download) or a plain directory tree.

Examples:
  curl -sL <hf>/data/train.tar | python -m retailaction.pack --src - --out train.pack.tar
  python -m retailaction.pack --src competition.zip --prefix challenge_test_set/ --out challenge.pack.tar
  (--src may also be an https URL of the zip, or @file holding that URL; members
  are then read with range requests)
"""
import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

from .remote import HttpRangeFile

SIZE = 256
REQUIRED = ("rank0_video.mp4", "rank1_video.mp4", "metadata.json")


def reencode(data: bytes, size: int = SIZE) -> bytes:
    cmd = [
        "ffmpeg", "-v", "error", "-threads", "1", "-f", "mp4", "-i", "pipe:0",
        "-vf", f"scale={size}:{size}:flags=area", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-threads", "1", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ]
    # mp4 input from a pipe needs the moov atom first; write to a temp file if that fails.
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode == 0 and p.stdout:
        return p.stdout
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        cmd[cmd.index("pipe:0")] = f.name
        p = subprocess.run(cmd, capture_output=True, check=True)
        return p.stdout


def _split(path):
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[-1].startswith("._"):
        return None, None
    return parts[-2], parts[-1]


def iter_samples(src, prefix="", ids=None):
    """Yield (sample id, {file name: bytes}) for every sample directory in the source."""
    if src == "-" or (os.path.isfile(src) and tarfile.is_tarfile(src)):
        fobj = sys.stdin.buffer if src == "-" else open(src, "rb")
        cur, files = None, {}
        with tarfile.open(fileobj=fobj, mode="r|*") as tf:
            try:
                for m in tf:
                    if not m.isfile():
                        continue
                    sid, name = _split(m.name)
                    if sid is None:
                        continue
                    if sid != cur and files:
                        yield cur, files
                        files = {}
                    cur = sid
                    files[name] = tf.extractfile(m).read()
            except (tarfile.ReadError, EOFError) as e:
                print(f"tar stream ended early: {e}", file=sys.stderr)
                files = {}  # last sample may be truncated
        if files:
            yield cur, files
    elif src.startswith("http") or zipfile.is_zipfile(src):
        fobj = HttpRangeFile(src) if src.startswith("http") else src
        with zipfile.ZipFile(fobj) as zf:
            groups = {}
            for info in zf.infolist():
                n = info.filename
                if not n.startswith(prefix) or n.endswith("/"):
                    continue
                sid, name = _split(n)
                if sid is not None and (ids is None or sid in ids):
                    groups.setdefault(sid, []).append((info.header_offset, name, info))
            # read in archive order so a remote source is fetched sequentially
            for sid in sorted(groups, key=lambda k: min(g[0] for g in groups[k])):
                yield sid, {name: zf.read(info) for _, name, info in sorted(groups[sid], key=lambda g: g[0])}
    else:
        for root, _, names in sorted(os.walk(src)):
            if names:
                files = {}
                for f in names:
                    with open(os.path.join(root, f), "rb") as fh:
                        files[f] = fh.read()
                yield os.path.basename(root), files


def process(sid, files):
    out = {}
    out["r0.mp4"] = reencode(files.pop("rank0_video.mp4"))
    out["r1.mp4"] = reencode(files.pop("rank1_video.mp4"))
    meta = json.loads(files["metadata.json"])
    for name, data in files.items():
        if name.endswith(".json"):
            out[name + ".gz"] = gzip.compress(data, 6)
    content = meta.get("content", meta) if isinstance(meta, dict) else {}
    labels = (content.get("labels") or {}).get("action")
    return sid, out, labels


def build_index(out_path):
    index = {}
    with tarfile.open(out_path, "r:") as tf:
        for m in tf:
            if m.isfile():
                sid, name = m.name.split("/", 1)
                index.setdefault(sid, {})[name] = [m.offset_data, m.size]
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many samples (0 = all)")
    ap.add_argument("--prefix", default="", help="zip sources: only members under this path")
    ap.add_argument("--ids", default="", help="optional text file with one sample id per line")
    args = ap.parse_args()

    labels_all, done = {}, 0
    t0 = time.time()
    futures = set()
    tf = tarfile.open(args.out, "w", format=tarfile.USTAR_FORMAT)

    def drain(block_all=False):
        nonlocal futures, done
        while futures and (block_all or len(futures) >= args.workers * 3):
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                sid, out, labels = fut.result()
                for name, data in out.items():
                    ti = tarfile.TarInfo(f"{sid}/{name}")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                labels_all[sid] = labels
                done += 1
                if done % 200 == 0:
                    print(f"{done} samples, {time.time() - t0:.0f}s", flush=True)

    skipped = []
    with ThreadPoolExecutor(args.workers) as pool:
        ids = set(open(args.ids).read().split()) if args.ids else None
        if args.src.startswith("@"):  # read a (signed) URL from a file so it stays out of logs
            args.src = open(args.src[1:]).read().strip()
        for sid, files in iter_samples(args.src, args.prefix, ids):
            if not all(r in files for r in REQUIRED):
                skipped.append(sid)
                continue
            futures.add(pool.submit(process, sid, files))
            drain()
            if args.limit and done + len(futures) >= args.limit:
                break
        drain(block_all=True)
    tf.close()
    if skipped:
        print(f"incomplete samples skipped ({len(skipped)}): {skipped[:20]}", file=sys.stderr)

    index = build_index(args.out)
    with open(args.out + ".index.json", "w") as f:
        json.dump({"samples": index, "labels": labels_all}, f)
    print(f"packed {len(index)} samples into {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
