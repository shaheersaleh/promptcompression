import argparse
import csv
import json
import torch
import torch.nn.functional as F
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
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_eval_tokens", type=int, default=16)
    parser.add_argument("--output_csv", type=str, default="token_distribution_trace.csv")
    return parser.parse_args()

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
    item = samples[args.sample_idx]
    prompt = item["prompt"]

    prefix, rest = prompt.split("Context: ", 1)
    context_str, suffix = rest.split("\n\nQuestion:", 1)
    prefix += "Context: "
    suffix = "\n\nQuestion:" + suffix

    # 1. Compress context
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
    p_comp = phi_tok(comp_prompt, return_tensors="pt").to(device)

    # 2. Generate ground-truth sequence from uncompressed prompt
    with torch.no_grad():
        out_orig = phi_model.generate(**p_orig, max_new_tokens=args.max_eval_tokens, do_sample=False)
    
    gen_token_ids = out_orig[0][p_orig["input_ids"].shape[-1]:].tolist()

    print("=" * 105)
    print(f"STEP-BY-STEP TOKEN DISTRIBUTION DIVERGENCE (Sample {args.sample_idx})")
    print(f"Original Length:   {p_orig['input_ids'].shape[-1]} tokens")
    print(f"Compressed Length: {p_comp['input_ids'].shape[-1]} tokens ({(1.0 - p_comp['input_ids'].shape[-1]/p_orig['input_ids'].shape[-1])*100:.1f}% dropped)")
    print("=" * 105)

    trace_records = []
    
    # 3. Prefill initial KV caches
    with torch.no_grad():
        orig_init = phi_model(**p_orig, use_cache=True)
        comp_init = phi_model(**p_comp, use_cache=True)

        orig_past = orig_init.past_key_values
        comp_past = comp_init.past_key_values

        orig_logits = orig_init.logits[:, -1, :].float()
        comp_logits = comp_init.logits[:, -1, :].float()

    print(f"{'Step':<5} | {'Gold Token':<15} | {'P(Gold)':<10} | {'Q(Gold)':<10} | {'Rank(Q)':<8} | {'Step KL':<10} | {'Top Alternative in Q':<20}")
    print("-" * 105)

    with torch.no_grad():
        for step_idx, gold_id in enumerate(gen_token_ids):
            p_probs = F.softmax(orig_logits, dim=-1).squeeze(0)
            q_probs = F.softmax(comp_logits, dim=-1).squeeze(0)

            log_p = F.log_softmax(orig_logits, dim=-1).squeeze(0)
            log_q = F.log_softmax(comp_logits, dim=-1).squeeze(0)
            step_kl = torch.sum(p_probs * (log_p - log_q)).item()

            p_gold = p_probs[gold_id].item()
            q_gold = q_probs[gold_id].item()

            # Rank of gold token under Q
            q_sorted_indices = torch.argsort(q_probs, descending=True)
            q_rank = (q_sorted_indices == gold_id).nonzero(as_tuple=True)[0].item() + 1

            q_top1_id = q_sorted_indices[0].item()
            q_top1_token = repr(phi_tok.decode([q_top1_id]))
            q_top1_prob = q_probs[q_top1_id].item()
            alt_info = f"{q_top1_token} ({q_top1_prob:.2f})" if q_top1_id != gold_id else "None (Exact Top-1)"

            gold_str = repr(phi_tok.decode([gold_id]))
            print(f"{step_idx+1:<5} | {gold_str:<15} | {p_gold:<10.4f} | {q_gold:<10.4f} | {q_rank:<8} | {step_kl:<10.4f} | {alt_info:<20}")

            trace_records.append({
                "Step": step_idx + 1,
                "Gold Token": phi_tok.decode([gold_id]),
                "P(Gold)": f"{p_gold:.4f}",
                "Q(Gold)": f"{q_gold:.4f}",
                "Rank in Q": q_rank,
                "Step KL Divergence": f"{step_kl:.4f}",
                "Top Q Prediction": phi_tok.decode([q_top1_id]),
                "Top Q Prob": f"{q_top1_prob:.4f}"
            })

            # Advance single step with KV cache
            step_input = torch.tensor([[gold_id]], device=device)
            orig_step_out = phi_model(input_ids=step_input, past_key_values=orig_past, use_cache=True)
            comp_step_out = phi_model(input_ids=step_input, past_key_values=comp_past, use_cache=True)

            orig_past = orig_step_out.past_key_values
            comp_past = comp_step_out.past_key_values

            orig_logits = orig_step_out.logits[:, -1, :].float()
            comp_logits = comp_step_out.logits[:, -1, :].float()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trace_records[0].keys())
        writer.writeheader()
        writer.writerows(trace_records)

    print("-" * 105)
    print(f"Exported detailed trace to {args.output_csv}")

if __name__ == "__main__":
    main()
