import json
import sys
import numpy as np

def load_audit_metrics(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    drop_rates = [r["token_drop_rate_pct"] for r in records]
    f1s = [r["metrics"]["token_f1"] for r in records]
    ems = [r["metrics"]["exact_match"] for r in records]
    
    max_steps = 15
    step_kls = {step: [] for step in range(1, max_steps + 1)}
    for r in records:
        for s in r.get("probability_audit_steps", []):
            st = s.get("step")
            if st in step_kls:
                step_kls[st].append(s.get("step_kl", 0.0))
                
    avg_step_kl = {st: float(np.mean(vals)) if vals else 0.0 for st, vals in step_kls.items()}

    return {
        "count": len(records),
        "mean_drop": float(np.mean(drop_rates)),
        "mean_f1": float(np.mean(f1s)),
        "mean_em": float(np.mean(ems)),
        "step_kl": avg_step_kl
    }

def print_comparison(baseline_path, traj_path):
    b = load_audit_metrics(baseline_path)
    t = load_audit_metrics(traj_path)

    print(f"\n{'Metric':<28} | {'Token-0 Base (epoch_1)':<24} | {'Trajectory RL (final)':<24}")
    print("-" * 82)
    print(f"{'Evaluated Samples':<28} | {b['count']:<24} | {t['count']:<24}")
    print(f"{'Mean Token Drop Rate (%)':<28} | {b['mean_drop']:<23.2f}% | {t['mean_drop']:<23.2f}%")
    print(f"{'Mean Downstream Token F1':<28} | {b['mean_f1']:<24.4f} | {t['mean_f1']:<24.4f}")
    print(f"{'Downstream Exact Match (EM)':<28} | {b['mean_em']:<24.4f} | {t['mean_em']:<24.4f}")
    print("-" * 82)
    print("\nStep-Wise KL Drift along Output Trajectory:")
    print(f"{'Token Horizon Step':<28} | {'Base Step KL (t=0 trained)':<24} | {'Trajectory RL Step KL':<24}")
    print("-" * 82)
    for step in range(1, 11):
        print(f"Token t={step:<20} | {b['step_kl'].get(step, 0.0):<24.4f} | {t['step_kl'].get(step, 0.0):<24.4f}")

if __name__ == "__main__":
    base_file = sys.argv[1] if len(sys.argv) > 1 else "comprehensive_audit_results.jsonl"
    traj_file = sys.argv[2] if len(sys.argv) > 2 else "audit_results_trajectory_rl.jsonl"
    print_comparison(base_file, traj_file)
