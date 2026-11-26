import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import requests
from sklearn import metrics as skm

CLASSES = ["uninfected", "parasitized"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a deployed model through its /predict endpoint.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--out", default="production_evaluation.json")
    args = parser.parse_args()

    rng = random.Random(42)
    items = []
    for label, name in enumerate(CLASSES):
        files = sorted((Path(args.test_dir) / name).glob("*.png"))
        items += [(p, label) for p in rng.sample(files, min(args.per_class, len(files)))]

    base = args.url.rstrip("/")
    y_true, y_prob, latencies, versions = [], [], [], set()
    for path, label in items:
        start = time.perf_counter()
        with open(path, "rb") as f:
            # A free instance can take a minute to wake up, so the first call needs a long timeout.
            response = requests.post(f"{base}/predict", files={"file": f}, timeout=180)
        response.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)
        body = response.json()
        y_true.append(label)
        y_prob.append(body["probability_parasitized"])
        versions.add(body["model_version"])

    y_pred = [int(p >= 0.5) for p in y_prob]
    result = {
        "url": base,
        "model_versions_seen": sorted(versions),
        "n_images": len(items),
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred),
        "recall": skm.recall_score(y_true, y_pred),
        "f1": skm.f1_score(y_true, y_pred),
        "roc_auc": skm.roc_auc_score(y_true, y_prob),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
