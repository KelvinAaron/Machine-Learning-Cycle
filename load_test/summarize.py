import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

results = Path(sys.argv[1] if len(sys.argv) > 1 else "load_test/results")
rows = []
for stats in sorted(results.glob("containers_*_stats.csv")):
    n = int(stats.name.split("_")[1])
    total = pd.read_csv(stats).query("Name == 'Aggregated'").iloc[0]
    rows.append(
        {
            "containers": n,
            "requests": int(total["Request Count"]),
            "failures": int(total["Failure Count"]),
            "requests_per_s": round(total["Requests/s"], 2),
            "median_ms": total["Median Response Time"],
            "p95_ms": total["95%"],
            "mean_ms": round(total["Average Response Time"], 1),
        }
    )
table = pd.DataFrame(rows).set_index("containers")
print(table)
table.to_csv(results / "summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
table[["median_ms", "p95_ms"]].plot(kind="bar", ax=axes[0], rot=0, title="Response time (ms)")
table["requests_per_s"].plot(kind="bar", ax=axes[1], rot=0, color="#5a3e8e", title="Throughput (requests/s)")
plt.tight_layout()
plt.savefig(results / "summary.png", dpi=150)
