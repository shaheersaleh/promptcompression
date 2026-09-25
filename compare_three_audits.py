import json
import sys
import numpy as np

def load_audit(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    drops = [r["token_drop_rate_pct"] for r in records]
    f1s = [r["metrics"]["token_f1"] for r in records]
    ems = [r["metrics"]["exact_match"] for r in records]
    
    step_kls = {s: [] for s in range(1, 11)}
    for r in records:
        for step_data in r.get("probability_audit_steps", []):
            st = step_data.get("step")
            if st in step_kls:
                step_kls[st].append(step_data.get("step_kl", 0.0))
                
    avg_step_kl = {s: float(np.mean(vals)) if vals else 0.0 for s, vals in step_kls.items()}
    
    return {
        "count": len(records),
        "mean_drop": float(np.mean(drops)),
        "mean_f1": float(np.mean(f1s)),
        "mean_em": float(np.mean(ems)),
        "step_kl": avg_step_kl
    }

b = load_audit("comprehensive_audit_results.jsonl")
t1 = load_audit("audit_results_trajectory_rl.jsonl")
t4 = load_audit("audit_results_ddp.jsonl")

print(f"\n{'Metric':<27} | {'Base (t=0)':<16} | {'Traj RL (1-GPU)':<18} | {'Traj DDP (4-GPU)':<18}")
print("-" * 88)
print(f"{'Evaluated Samples':<27} | {b['count']:<16} | {t1['count']:<18} | {t4['count']:<18}")
print(f"{'Mean Token Drop Rate (%)':<27} | {b['mean_drop']:<15.2f}% | {t1['mean_drop']:<17.2f}% | {t4['mean_drop']:<17.2f}%")
print(f"{'Mean Downstream Token F1':<27} | {b['mean_f1']:<16.4f} | {t1['mean_f1']:<18.4f} | {t4['mean_f1']:<18.4f}")
print(f"{'Downstream Exact Match':<27} | {b['mean_em']:<16.4f} | {t1['mean_em']:<18.4f} | {t4['mean_em']:<18.4f}")
print("-" * 88)
print("\nStep-Wise KL Drift along the Output Trajectory:")
print(f"{'Horizon Step':<27} | {'Base (t=0)':<16} | {'Traj RL (1-GPU)':<18} | {'Traj DDP (4-GPU)':<18}")
print("-" * 88)
for s in range(1, 11):
    print(f"Token t={s:<20} | {b['step_kl'][s]:<16.4f} | {t1['step_kl'][s]:<18.4f} | {t4['step_kl'][s]:<18.4f}")
