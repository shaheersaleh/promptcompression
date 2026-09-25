import argparse
import glob
import json
import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints with single Phi-4 load.")
    parser.add_argument("--phi4_id", type=str, default="microsoft/phi-4")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Base checkpoints dir")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--num_eval_samples", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=1536)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load test data
    samples = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    eval_samples = samples[-args.num_eval_samples:] if len(samples) >= args.num_eval_samples else samples
    print(f"Loaded {len(eval_samples)} test samples.")

    # 2. Load Phi-4 ONCE
    print("Loading Phi-4 in 4-bit (single load for all evaluations)...")
    phi_tok = AutoTokenizer.from_pretrained(args.phi4_id, trust_remote_code=True)
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

    # Pre-tokenize original prompts to avoid recomputing baseline outputs
    print("Precomputing original uncompressed baseline outputs from Phi-4...")
    orig_cache = []
    with torch.no_grad():
        for item in eval_samples:
            prompt = item["prompt"]
            inputs = phi_tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            out = phi_model(**inputs).logits[:, -1, :].float()
            p = F.softmax(out, dim=-1)
            log_p = F.log_softmax(out, dim=-1)
            top1 = torch.argmax(out, dim=-1).item()
            top5 = torch.topk(out, k=5, dim=-1).indices.squeeze(0).tolist()
            orig_cache.append({
                "prompt": prompt,
                "n_tokens": inputs["input_ids"].shape[-1],
                "p": p.cpu(),
                "log_p": log_p.cpu(),
                "top1": top1,
                "top5": top5
            })

    # 3. Find and sort all checkpoint directories
    ckpt_dirs = sorted(glob.glob(os.path.join(args.checkpoints_dir, "checkpoint_epoch_*")))
    if not ckpt_dirs:
        print(f"No checkpoint_epoch_* folders found in {args.checkpoints_dir}")
        return

    print(f"\nFound checkpoints to evaluate: {[os.path.basename(c) for c in ckpt_dirs]}")

    # 4. Evaluate each compressor checkpoint
    for ckpt_path in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt_path)
        print("\n" + "=" * 60)
        print(f"EVALUATING: {ckpt_name}")
        print("=" * 60)

        enc_tok = AutoTokenizer.from_pretrained(ckpt_path)
        compressor = AutoModelForTokenClassification.from_pretrained(ckpt_path).to(device)
        compressor.eval()

        total_orig_tokens = 0
        total_comp_tokens = 0
        total_kl = 0.0
        top1_matches = 0
        top5_matches = 0

        with torch.no_grad():
            for i, (item, base) in enumerate(zip(eval_samples, orig_cache)):
                prompt = item["prompt"]

                if "Context: " in prompt and "\n\nQuestion:" in prompt:
                    prefix, rest = prompt.split("Context: ", 1)
                    context_str, suffix = rest.split("\n\nQuestion:", 1)
                    prefix += "Context: "
                    suffix = "\n\nQuestion:" + suffix
                else:
                    prefix, context_str, suffix = "", prompt, ""

                enc_inputs = enc_tok(context_str, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
                enc_logits = compressor(**enc_inputs).logits.squeeze(0)
                keep_probs = torch.softmax(enc_logits, dim=-1)[:, 1]

                keep_mask = (keep_probs >= args.threshold).long()
                if keep_mask.sum() == 0:
                    keep_mask[torch.argmax(keep_probs)] = 1

                input_tokens = enc_tok.convert_ids_to_tokens(enc_inputs["input_ids"].squeeze(0))
                kept_tokens = [tok for tok, km in zip(input_tokens, keep_mask) if km.item() == 1]
                compressed_context = enc_tok.convert_tokens_to_string(kept_tokens).strip() or context_str
                full_compressed = prefix + compressed_context + suffix

                phi_comp_inputs = phi_tok(full_compressed, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
                n_comp = phi_comp_inputs["input_ids"].shape[-1]
                n_orig = base["n_tokens"]
                total_orig_tokens += n_orig
                total_comp_tokens += n_comp

                out_comp = phi_model(**phi_comp_inputs).logits[:, -1, :].float()
                log_q = F.log_softmax(out_comp, dim=-1).cpu()

                p = base["p"]
                log_p = base["log_p"]
                kl = torch.sum(p * (log_p - log_q)).item()
                total_kl += kl

                comp_top1 = torch.argmax(out_comp, dim=-1).item()
                if comp_top1 == base["top1"]:
                    top1_matches += 1
                if comp_top1 in base["top5"]:
                    top5_matches += 1

                if i < 2:
                    print(f"[{ckpt_name} Sample {i+1}] Orig: {n_orig} toks | Comp: {n_comp} toks ({(1.0 - n_comp/n_orig)*100:.1f}% drop) | KL: {kl:.4f}")

        avg_drop = (1.0 - (total_comp_tokens / total_orig_tokens)) * 100.0
        avg_kl = total_kl / len(eval_samples)
        top1_acc = (top1_matches / len(eval_samples)) * 100.0
        top5_acc = (top5_matches / len(eval_samples)) * 100.0

        print(f"\n--- {ckpt_name} RESULTS ---")
        print(f"  Average Token Drop Rate: {avg_drop:.2f}%")
        print(f"  Average KL Divergence:   {avg_kl:.4f}")
        print(f"  Top-1 Match Rate:        {top1_acc:.2f}%")
        print(f"  Top-5 Match Rate:        {top5_acc:.2f}%")

        del compressor, enc_tok
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
