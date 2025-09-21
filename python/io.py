# io.py
import json, csv, os
from dataclasses import asdict

def save_csv(path, times, marks):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "mark"])
        for t, v in zip(times, marks):
            w.writerow([f"{t:.12g}", f"{v:.12g}"])

def save_json(path, meta: dict, times, marks):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = dict(meta)
    obj["events"] = [{"t": float(t), "v": float(v)} for t, v in zip(times, marks)]
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
