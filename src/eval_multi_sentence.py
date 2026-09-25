import argparse
import csv
import json
import re
import string
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi4_id", type=str, default="microsoft/phi-4")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/checkpoint_epoch_1")
    parser.add_argument("--test_data", type=str, default="data/train.jsonl")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=35)
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv")
    parser.add_argument("--output_md", type=str, default="evaluation_results.md")
    return parser.parse_args()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1(pred, target):
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()
    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2 * (precision * recall) / (precision + recall)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phi_tok = AutoTokenizer.from_pretrained(args.phi4_id, trust_remote_code=True)
    enc_tok = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    phi_model = AutoModelForCausalLM.from_pretrained(
        args.phi4_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    phi_model.eval()

    compressor = AutoModelForTokenClassification.from_pretrained(args.checkpoint_dir).to(device)
    compressor.eval()

    with open(args.test_data, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    eval_slice = samples[:args.num_samples]
    results = []

    print(f"Benchmarking {len(eval_slice)} samples and exporting tables...")

    for i, item in enumerate(eval_slice):
        prompt = item["prompt"]

        if "Context: " in prompt and "\n\nQuestion:" in prompt:
            prefix, rest = prompt.split("Context: ", 1)
            context_str, suffix = rest.split("\n\nQuestion:", 1)
            prefix += "Context: "
            question_str = suffix.split("<|im_end|>")[0].replace("\n\nQuestion:", "").strip()
            suffix = "\n\nQuestion:" + suffix
        else:
            prefix, context_str, suffix = "", prompt, ""
            question_str = "N/A"

        with torch.no_grad():
            enc_inputs = enc_tok(context_str, return_tensors="pt", truncation=True, max_length=1536).to(device)
            enc_logits = compressor(**enc_inputs).logits.squeeze(0)
            keep_mask = (torch.softmax(enc_logits, dim=-1)[:, 1] >= 0.5).long()
            if keep_mask.sum() == 0:
                keep_mask[torch.argmax(enc_logits[:, 1])] = 1

            input_tokens = enc_tok.convert_ids_to_tokens(enc_inputs["input_ids"].squeeze(0))
            kept = [tok for tok, m in zip(input_tokens, keep_mask) if m.item() == 1]
            comp_context = enc_tok.convert_tokens_to_string(kept).strip() or context_str
            comp_prompt = prefix + comp_context + suffix

            p_orig = phi_tok(prompt, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            out_orig = phi_model.generate(**p_orig, max_new_tokens=args.max_new_tokens, do_sample=False)
            orig_latency = time.perf_counter() - t0
            ans_orig = phi_tok.decode(out_orig[0][p_orig["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            p_comp = phi_tok(comp_prompt, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            out_comp = phi_model.generate(**p_comp, max_new_tokens=args.max_new_tokens, do_sample=False)
            comp_latency = time.perf_counter() - t0
            ans_comp = phi_tok.decode(out_comp[0][p_comp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        orig_tokens = p_orig["input_ids"].shape[-1]
        comp_tokens = p_comp["input_ids"].shape[-1]
        drop_rate = (1.0 - comp_tokens / orig_tokens) * 100.0
        f1 = compute_f1(ans_comp, ans_orig)
        em = int(normalize_answer(ans_comp) == normalize_answer(ans_orig))

        results.append({
            "Sample": i + 1,
            "Question": question_str,
            "Orig Tokens": orig_tokens,
            "Comp Tokens": comp_tokens,
            "Drop Rate (%)": f"{drop_rate:.1f}%",
            "F1 Score": f"{f1:.2f}",
            "Exact Match": em,
            "Orig Latency (s)": f"{orig_latency:.2f}",
            "Comp Latency (s)": f"{comp_latency:.2f}",
            "Original Answer": ans_orig.replace("\n", " "),
            "Compressed Answer": ans_comp.replace("\n", " ")
        })

    # Write CSV
    keys = results[0].keys()
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    # Write Markdown Table
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Downstream Generation Performance\n\n")
        f.write("| ID | Question | Orig Tok | Comp Tok | Drop % | F1 | EM | Orig Ans | Comp Ans |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            q_trunc = (r['Question'][:30] + '...') if len(r['Question']) > 30 else r['Question']
            o_ans = (r['Original Answer'][:40] + '...') if len(r['Original Answer']) > 40 else r['Original Answer']
            c_ans = (r['Compressed Answer'][:40] + '...') if len(r['Compressed Answer']) > 40 else r['Compressed Answer']
            f.write(f"| {r['Sample']} | {q_trunc} | {r['Orig Tokens']} | {r['Comp Tokens']} | {r['Drop Rate (%)']} | {r['F1 Score']} | {r['Exact Match']} | {o_ans} | {c_ans} |\n")

    print(f"Generated {args.output_csv} and {args.output_md}")

if __name__ == "__main__":
    main()
