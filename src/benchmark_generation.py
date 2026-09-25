import argparse
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
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
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

    samples = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    eval_samples = samples[-args.num_samples:]

    total_f1 = 0.0
    exact_matches = 0
    orig_time_total = 0.0
    comp_time_total = 0.0

    print(f"Benchmarking generation over {len(eval_samples)} samples...\n")

    with torch.no_grad():
        for idx, item in enumerate(eval_samples):
            prompt = item["prompt"]
            if "Context: " in prompt and "\n\nQuestion:" in prompt:
                prefix, rest = prompt.split("Context: ", 1)
                context_str, suffix = rest.split("\n\nQuestion:", 1)
                prefix += "Context: "
                suffix = "\n\nQuestion:" + suffix
            else:
                prefix, context_str, suffix = "", prompt, ""

            # Compress context
            enc_inputs = enc_tok(context_str, return_tensors="pt", truncation=True, max_length=1536).to(device)
            enc_logits = compressor(**enc_inputs).logits.squeeze(0)
            keep_probs = torch.softmax(enc_logits, dim=-1)[:, 1]
            keep_mask = (keep_probs >= args.threshold).long()
            if keep_mask.sum() == 0:
                keep_mask[torch.argmax(keep_probs)] = 1

            input_tokens = enc_tok.convert_ids_to_tokens(enc_inputs["input_ids"].squeeze(0))
            kept = [tok for tok, m in zip(input_tokens, keep_mask) if m.item() == 1]
            compressed_context = enc_tok.convert_tokens_to_string(kept).strip() or context_str
            comp_prompt = prefix + compressed_context + suffix

            # 1. Generate from uncompressed prompt
            p_orig = phi_tok(prompt, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            out_orig = phi_model.generate(**p_orig, max_new_tokens=args.max_new_tokens, do_sample=False)
            t_orig = time.perf_counter() - t0
            orig_time_total += t_orig
            ans_orig = phi_tok.decode(out_orig[0][p_orig["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            # 2. Generate from compressed prompt
            p_comp = phi_tok(comp_prompt, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            out_comp = phi_model.generate(**p_comp, max_new_tokens=args.max_new_tokens, do_sample=False)
            t_comp = time.perf_counter() - t0
            comp_time_total += t_comp
            ans_comp = phi_tok.decode(out_comp[0][p_comp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            f1 = compute_f1(ans_comp, ans_orig)
            em = int(normalize_answer(ans_comp) == normalize_answer(ans_orig))
            total_f1 += f1
            exact_matches += em

            if idx < 3:
                print(f"--- Sample {idx+1} ---")
                print(f"Uncompressed ({p_orig['input_ids'].shape[-1]} toks): '{ans_orig}'")
                print(f"Compressed   ({p_comp['input_ids'].shape[-1]} toks): '{ans_comp}'")
                print(f"Agreement: EM={em} | F1={f1:.4f}\n")

    print("=" * 50)
    print(f"GENERATION AGREEMENT (Compressed vs Original):")
    print(f"  Exact Match (EM): {exact_matches / len(eval_samples) * 100:.2f}%")
    print(f"  Token F1 Score:   {total_f1 / len(eval_samples) * 100:.2f}%")
    print(f"  Total Orig Prefill+Gen Time: {orig_time_total:.2f}s")
    print(f"  Total Comp Prefill+Gen Time: {comp_time_total:.2f}s ({(1 - comp_time_total/orig_time_total)*100:.1f}% faster)")
    print("=" * 50)

if __name__ == "__main__":
    main()
