import glob
import json
import os
import re
from typing import Any, Dict

def get_latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        return ""
    return max(files, key=os.path.getmtime)

def parse_audit_log(filepath: str) -> Dict[int, Dict[str, Any]]:
    """Parses token reduction, target sentence, and likelihood ratios from token_audit logs."""
    data = {}
    if not filepath or not os.path.exists(filepath):
        return data

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    sample_blocks = re.split(r"={40,}", content)
    for block in sample_blocks:
        sample_match = re.search(
            r"SAMPLE\s+(\d+)\s*\|\s*Prompt Reduction:\s*(\d+)\s*toks\s*->\s*(\d+)\s*toks\s*\(([\d\.]+)%\s*dropped\)",
            block
        )
        if not sample_match:
            continue

        sample_id = int(sample_match.group(1))
        orig_toks = int(sample_match.group(2))
        comp_toks = int(sample_match.group(3))
        drop_pct = float(sample_match.group(4))

        target_match = re.search(r'Target Sentence:\s*"([^"]+)"', block)
        target_sent = target_match.group(1) if target_match else ""

        ratio_match = re.search(r"Likelihood Ratio \(Q\s*/\s*P\):\s*([\d\.]+)", block)
        likelihood_ratio = float(ratio_match.group(1)) if ratio_match else None

        data[sample_id] = {
            "sample_id": sample_id,
            "target_sentence": target_sent,
            "uncompressed_tokens": orig_toks,
            "compressed_tokens": comp_toks,
            "drop_rate_pct": drop_pct,
            "likelihood_ratio_q_over_p": likelihood_ratio,
        }
    return data

def parse_eval_log(filepath: str) -> Dict[int, Dict[str, Any]]:
    """Parses questions, contexts, completions, and QA metrics from eval_tokens logs."""
    data = {}
    if not filepath or not os.path.exists(filepath):
        return data

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split by sample indicators or dividers
    blocks = re.split(r"(?:={40,}|-{40,}|SAMPLE\s+\d+)", content)
    for block in blocks:
        id_match = re.search(r"(?:Sample|ID)\s*[:#]?\s*(\d+)", block, re.IGNORECASE)
        if not id_match:
            continue
        sample_id = int(id_match.group(1))

        # Extract Question / Query
        q_match = re.search(r"(?:Question|Query)\s*:\s*(.+?)(?=\n[A-Z]|\n\n|$)", block, re.DOTALL | re.IGNORECASE)
        # Extract Original / Uncompressed Context & Answer
        uncomp_ctx = re.search(r"(?:Uncompressed Context|Original Context)\s*:\s*(.+?)(?=\n[A-Z]|\n\n|$)", block, re.DOTALL | re.IGNORECASE)
        uncomp_ans = re.search(r"(?:Uncompressed Answer|Original Answer|Baseline Answer)\s*:\s*(.+?)(?=\n[A-Z]|\n\n|$)", block, re.DOTALL | re.IGNORECASE)

        # Extract Compressed Context & Answer
        comp_ctx = re.search(r"Compressed Context\s*:\s*(.+?)(?=\n[A-Z]|\n\n|$)", block, re.DOTALL | re.IGNORECASE)
        comp_ans = re.search(r"Compressed Answer\s*:\s*(.+?)(?=\n[A-Z]|\n\n|$)", block, re.DOTALL | re.IGNORECASE)

        # Extract F1 and EM if printed
        f1_match = re.search(r"(?:Token\s*)?F1\s*[:=]\s*([\d\.]+)", block, re.IGNORECASE)
        em_match = re.search(r"(?:Exact Match|EM)\s*[:=]\s*([01])", block, re.IGNORECASE)

        data[sample_id] = {
            "sample_id": sample_id,
            "question": q_match.group(1).strip() if q_match else "",
            "uncompressed_context": uncomp_ctx.group(1).strip() if uncomp_ctx else "",
            "uncompressed_answer": uncomp_ans.group(1).strip() if uncomp_ans else "",
            "compressed_context": comp_ctx.group(1).strip() if comp_ctx else "",
            "compressed_answer": comp_ans.group(1).strip() if comp_ans else "",
            "f1": float(f1_match.group(1)) if f1_match else None,
            "em": int(em_match.group(1)) if em_match else None,
        }
    return data

def main():
    audit_file = get_latest_file("logs/token_audit-*.out")
    eval_file = get_latest_file("logs/eval_tokens-*.out")

    print(f"[+] Parsing Audit Log: {audit_file or 'Not found'}")
    print(f"[+] Parsing Eval Log:  {eval_file or 'Not found'}")

    audit_data = parse_audit_log(audit_file)
    eval_data = parse_eval_log(eval_file)

    all_ids = sorted(set(audit_data.keys()) | set(eval_data.keys()))
    if not all_ids:
        print("[!] No sample records could be parsed. Check your log file paths and patterns.")
        return

    merged_records = []
    print("\n" + "=" * 110)
    print(f"{'ID':<4} | {'Drop %':<8} | {'Q/P Ratio':<10} | {'F1':<6} | {'Target Query / Compressed Answer'}")
    print("=" * 110)

    for sid in all_ids:
        a_rec = audit_data.get(sid, {})
        e_rec = eval_data.get(sid, {})

        record = {
            "sample_id": sid,
            "query": e_rec.get("question") or a_rec.get("target_sentence", ""),
            "uncompressed_context": e_rec.get("uncompressed_context", ""),
            "uncompressed_answer": e_rec.get("uncompressed_answer", ""),
            "compressed_context": e_rec.get("compressed_context", ""),
            "compressed_answer": e_rec.get("compressed_answer", ""),
            "uncompressed_tokens": a_rec.get("uncompressed_tokens"),
            "compressed_tokens": a_rec.get("compressed_tokens"),
            "drop_rate_pct": a_rec.get("drop_rate_pct"),
            "likelihood_ratio": a_rec.get("likelihood_ratio_q_over_p"),
            "f1": e_rec.get("f1"),
            "em": e_rec.get("em"),
        }
        merged_records.append(record)

        drop_str = f"{record['drop_rate_pct']:.1f}%" if record['drop_rate_pct'] is not None else "N/A"
        ratio_str = f"{record['likelihood_ratio']:.4f}" if record['likelihood_ratio'] is not None else "N/A"
        f1_str = f"{record['f1']:.2f}" if record['f1'] is not None else "N/A"
        display_text = record["compressed_answer"] or record["query"]

        print(f"{sid:<4} | {drop_str:<8} | {ratio_str:<10} | {f1_str:<6} | {display_text[:65]}")

    output_path = "parsed_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, indent=2, ensure_ascii=False)

    print("=" * 110)
    print(f"[✓] Successfully parsed {len(merged_records)} samples -> Saved to '{output_path}'")

if __name__ == "__main__":
    main()
