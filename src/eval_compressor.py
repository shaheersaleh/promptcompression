import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate prompt compressor with Phi-4.")
    parser.add_argument("--phi4_id", type=str, default="microsoft/phi-4")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading evaluated checkpoint from {args.checkpoint_dir}...")

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

    total_tokens_orig = 0
    total_tokens_comp = 0
    total_kl = 0.0
    top1_matches = 0
    top5_matches = 0

    print(f"Evaluating {len(samples)} samples...\n")

    with torch.no_grad():
        for i, item in enumerate(samples):
            prompt = item["prompt"]

            # 1. Compress prompt body
            enc_inputs = enc_tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            enc_logits = compressor(**enc_inputs).logits.squeeze(0)
            keep_probs = torch.softmax(enc_logits, dim=-1)[:, 1]

            keep_mask = (keep_probs >= args.threshold).long()
            if keep_mask.sum() == 0:
                keep_mask[torch.argmax(keep_probs)] = 1

            input_tokens = enc_tok.convert_ids_to_tokens(enc_inputs["input_ids"].squeeze(0))
            kept_tokens = [tok for tok, km in zip(input_tokens, keep_mask) if km.item() == 1]
            compressed_prompt = enc_tok.convert_tokens_to_string(kept_tokens).strip() or prompt

            # Enforce structural suffix to mirror training
            if "Answer:" in prompt and not compressed_prompt.endswith("Answer:"):
                compressed_prompt += "\nAnswer:"

            # 2. Token counts under Phi-4's tokenizer
            phi_orig_inputs = phi_tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            phi_comp_inputs = phi_tok(compressed_prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

            n_orig = phi_orig_inputs["input_ids"].shape[-1]
            n_comp = phi_comp_inputs["input_ids"].shape[-1]
            total_tokens_orig += n_orig
            total_tokens_comp += n_comp

            # 3. Next-token distribution after 'Answer:'
            out_orig = phi_model(**phi_orig_inputs).logits[:, -1, :].float()
            out_comp = phi_model(**phi_comp_inputs).logits[:, -1, :].float()

            p = F.softmax(out_orig, dim=-1)
            log_p = F.log_softmax(out_orig, dim=-1)
            log_q = F.log_softmax(out_comp, dim=-1)
            kl = torch.sum(p * (log_p - log_q)).item()
            total_kl += kl

            orig_top1 = torch.argmax(out_orig, dim=-1).item()
            comp_top1 = torch.argmax(out_comp, dim=-1).item()
            top5_indices = torch.topk(out_orig, k=5, dim=-1).indices.squeeze(0).tolist()

            if orig_top1 == comp_top1:
                top1_matches += 1
            if comp_top1 in top5_indices:
                top5_matches += 1

            orig_token_str = phi_tok.decode([orig_top1])
            comp_token_str = phi_tok.decode([comp_top1])

            print(f"--- Sample {i+1} ---")
            print(f"[Original Length]: {n_orig} tokens -> Next predicted token: '{orig_token_str}'")
            print(f"[Compressed Length]: {n_comp} tokens ({(1.0 - n_comp/n_orig)*100:.1f}% reduction) -> Next predicted token: '{comp_token_str}'")
            print(f"[KL Divergence]: {kl:.4f}")
            print(f"[Compressed Prompt Text]:\n{compressed_prompt}\n")
            print("-" * 50)

    avg_drop = (1.0 - (total_tokens_comp / total_tokens_orig)) * 100.0
    avg_kl = total_kl / len(samples)
    top1_acc = (top1_matches / len(samples)) * 100.0
    top5_acc = (top5_matches / len(samples)) * 100.0

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS:")
    print(f"  Average Token Drop Rate: {avg_drop:.2f}%")
    print(f"  Average KL Divergence:   {avg_kl:.4f}")
    print(f"  Top-1 Distribution Match: {top1_acc:.2f}%")
    print(f"  Top-5 Distribution Match: {top5_acc:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
