import json
import os
import numpy as np

files = {
    "Base (t=0)": "comprehensive_audit_results.jsonl",
    "1-GPU (300)": "audit_results_trajectory_rl.jsonl",
    "4-GPU (300)": "audit_results_ddp.jsonl",
    "4-GPU (87k)": "audit_results_ddp_full.jsonl"
}

data = {}
for name, path in files.items():
    if not os.path.exists(path):
        print(f"[-] Note: {path} not found yet (audit may still be running).")
        continue
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not records:
        continue
    data[name] = {
        "drop": float(np.mean([r["token_drop_rate_pct"] for r in records])),
        "f1": float(np.mean([r["metrics"]["token_f1"] for r in records])),
        "em": float(np.mean([r["metrics"]["exact_match"] for r in records])),
        "kl": {s: float(np.mean([st.get("step_kl", 0.0) for r in records for st in r.get("probability_audit_steps", []) if st.get("step") == s])) for s in range(1, 11)}
    }

if not data:
    print("[!] No completed audit files found to compare.")
    exit(0)

line_len = 28 + len(data) * 16
metric_label = "Evaluation Metric"
header = f"\n{'=' * line_len}\n{metric_label:<25} | " + " | ".join([f"{n:<13}" for n in data])
print(header)
print("-" * line_len)
print(f"{'Mean Token Drop Rate (%)':<25} | " + " | ".join([f"{d['drop']:>12.2f}%" for d in data.values()]))
print(f"{'Mean Downstream F1 Score':<25} | " + " | ".join([f"{d['f1']:>13.4f}" for d in data.values()]))
print(f"{'Downstream Exact Match':<25} | " + " | ".join([f"{d['em']:>13.4f}" for d in data.values()]))
print("-" * line_len)

for s in [1, 2, 3, 4, 5, 8, 10]:
    row = " | ".join([f"{d['kl'].get(s, 0.0):>13.4f}" for d in data.values()])
    print(f"Step KL t={s:<17} | {row}")
print("=" * line_len + "\n")
