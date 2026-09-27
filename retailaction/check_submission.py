"""Check a submission CSV against sample_submission.csv and the documented format."""
import argparse
import csv
import json
import sys

csv.field_size_limit(1 << 30)


def check(path, sample):
    errors = []
    with open(sample, newline="") as f:
        ref = [r["id"] for r in csv.DictReader(f)]
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["id", "prediction"]:
            errors.append(f"columns are {r.fieldnames}, expected ['id', 'prediction']")
        rows = list(r)
    ids = [x["id"] for x in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    if set(ids) != set(ref):
        errors.append(f"id sets differ: {len(set(ref) - set(ids))} missing, {len(set(ids) - set(ref))} extra")
    n_act = 0
    for x in rows:
        try:
            p = json.loads(x["prediction"])
        except json.JSONDecodeError as e:
            errors.append(f"{x['id']}: bad JSON ({e})")
            continue
        acts = p.get("actions")
        if not isinstance(acts, list):
            errors.append(f"{x['id']}: 'actions' is not a list")
            continue
        for a in acts:
            n_act += 1
            try:
                vals = [a["score"], a["start"], a["end"]]
                for v in ("rank0", "rank1"):
                    vals += [a["spatial"]["action_cam"][v]["x"], a["spatial"]["action_cam"][v]["y"]]
            except (KeyError, TypeError):
                errors.append(f"{x['id']}: missing field in {a}")
                continue
            if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in vals):
                errors.append(f"{x['id']}: value out of [0, 1]")
            if not a["start"] < a["end"]:
                errors.append(f"{x['id']}: start >= end")
    return errors, len(rows), n_act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--sample", required=True)
    args = ap.parse_args()
    errors, n, n_act = check(args.submission, args.sample)
    print(f"{n} rows, {n_act} actions, {len(errors)} errors")
    for e in errors[:20]:
        print("  " + e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
