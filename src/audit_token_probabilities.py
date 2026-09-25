import argparse
import json
import math
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
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizers...", flush=True)
    phi_tok = AutoTokenizer.from_pretrained(args.phi4_id, trust_remote_code=True)
    enc_tok = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("Loading Phi-4 in 4-bit NF4...", flush=True)
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

    for sample_i, item in enumerate(eval_slice):
        prompt = item["prompt"]

        prefix, rest = prompt.split("Context: ", 1)
        context_str, suffix = rest.split("\n\nQuestion:", 1)
        prefix += "Context: "
        suffix = "\n\nQuestion:" + suffix

        # Compress context
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

        # Generate target sequence from uncompressed prompt
        with torch.no_grad():
            out_orig = phi_model.generate(**p_orig, max_new_tokens=args.max_tokens, do_sample=False)
        target_ids = out_orig[0][p_orig["input_ids"].shape[-1]:].tolist()

        # Prefill KV caches
        with torch.no_grad():
            orig_init = phi_model(**p_orig, use_cache=True)
            comp_init = phi_model(**p_comp, use_cache=True)

            orig_past = orig_init.past_key_values
            comp_past = comp_init.past_key_values

            orig_logits = orig_init.logits[:, -1, :].float()
            comp_logits = comp_init.logits[:, -1, :].float()

        cum_log_p = 0.0
        cum_log_q = 0.0

        n_orig = p_orig["input_ids"].shape[-1]
        n_comp = p_comp["input_ids"].shape[-1]
        drop_pct = (1.0 - n_comp / n_orig) * 100.0

        print("\n" + "=" * 115, flush=True)
        print(f"SAMPLE {sample_i + 1} | Prompt Reduction: {n_orig} toks -> {n_comp} toks ({drop_pct:.1f}% dropped)", flush=True)
        print(f"Target Sentence: \"{phi_tok.decode(target_ids, skip_special_tokens=True).strip()}\"", flush=True)
        print("=" * 115, flush=True)
        print(f"{'Step':<5} | {'Token':<16} | {'P(t)':<10} | {'Cumul P':<12} | {'Q(t)':<10} | {'Cumul Q':<12} | {'Rank(Q)':<8} | {'Step KL':<8}", flush=True)
        print("-" * 115, flush=True)

        with torch.no_grad():
            for step_idx, tid in enumerate(target_ids):
                p_dist = F.softmax(orig_logits, dim=-1).squeeze(0)
                q_dist = F.softmax(comp_logits, dim=-1).squeeze(0)

                log_p_dist = F.log_softmax(orig_logits, dim=-1).squeeze(0)
                log_q_dist = F.log_softmax(comp_logits, dim=-1).squeeze(0)
                step_kl = torch.sum(p_dist * (log_p_dist - log_q_dist)).item()

                p_val = p_dist[tid].item()
                q_val = q_dist[tid].item()

                cum_log_p += math.log(max(p_val, 1e-12))
                cum_log_q += math.log(max(q_val, 1e-12))

                cum_p = math.exp(cum_log_p)
                cum_q = math.exp(cum_log_q)

                # Fast O(1) rank calculation without sorting 100k vocabulary
                q_rank = (q_dist > q_val).sum().item() + 1
                token_label = repr(phi_tok.decode([tid]))

                print(f"{step_idx+1:<5} | {token_label:<16} | {p_val:<10.4f} | {cum_p:<12.4e} | {q_val:<10.4f} | {cum_q:<12.4e} | {q_rank:<8} | {step_kl:<8.4f}", flush=True)

                step_in = torch.tensor([[tid]], device=device)
                o_step = phi_model(input_ids=step_in, past_key_values=orig_past, use_cache=True)
                c_step = phi_model(input_ids=step_in, past_key_values=comp_past, use_cache=True)

                orig_past = o_step.past_key_values
                comp_past = c_step.past_key_values

                orig_logits = o_step.logits[:, -1, :].float()
                comp_logits = c_step.logits[:, -1, :].float()

        print("-" * 115, flush=True)
        print(f"Overall Sequence Likelihood P (Uncompressed): {math.exp(cum_log_p):.4e}  (log-prob: {cum_log_p:.2f})", flush=True)
        print(f"Overall Sequence Likelihood Q (Compressed):   {math.exp(cum_log_q):.4e}  (log-prob: {cum_log_q:.2f})", flush=True)
        print(f"Likelihood Ratio (Q / P):                     {math.exp(cum_log_q - cum_log_p):.4f}", flush=True)

if __name__ == "__main__":
    main()
