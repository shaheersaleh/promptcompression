import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument("--sample_idx", type=int, default=2)  # Sample 3 from bench (Houston, Texas)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output_plot", type=str, default="distribution_comparison.png")
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

        out_orig = phi_model(**p_orig).logits[:, -1, :].float()
        out_comp = phi_model(**p_comp).logits[:, -1, :].float()

        probs_orig = F.softmax(out_orig, dim=-1).squeeze(0)
        probs_comp = F.softmax(out_comp, dim=-1).squeeze(0)

        top_probs_orig, top_indices = torch.topk(probs_orig, k=args.top_k)
        top_probs_comp = probs_comp[top_indices]

    n_orig = p_orig["input_ids"].shape[-1]
    n_comp = p_comp["input_ids"].shape[-1]
    kl = torch.sum(probs_orig * (F.log_softmax(out_orig, dim=-1) - F.log_softmax(out_comp, dim=-1))).item()

    print(f"\n--- Distribution Alignment for Sample {args.sample_idx} ---")
    print(f"Original Length:   {n_orig} tokens")
    print(f"Compressed Length: {n_comp} tokens ({(1.0 - n_comp/n_orig)*100:.1f}% reduction)")
    print(f"KL Divergence:     {kl:.4f}\n")
    print(f"{'Rank':<5} | {'Token':<16} | {'P(Uncompressed)':<18} | {'Q(Compressed)':<18} | {'Diff'}")
    print("-" * 70)
    tokens = []
    for rank, (idx, po, qc) in enumerate(zip(top_indices, top_probs_orig, top_probs_comp), 1):
        raw_token = phi_tok.decode([idx.item()])
        token_label = repr(raw_token)
        tokens.append(raw_token.strip() or token_label)
        diff = qc.item() - po.item()
        print(f"{rank:<5} | {token_label:<16} | {po.item():<18.4f} | {qc.item():<18.4f} | {diff:+.4f}")
    print("-" * 70)

    # Generate Plot
    p_vals = top_probs_orig.cpu().numpy()
    q_vals = top_probs_comp.cpu().numpy()
    x = np.arange(len(tokens))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, p_vals, width, label='Uncompressed Prompt (P)', color='#2b5c8f')
    plt.bar(x + width/2, q_vals, width, label='Compressed Context (Q)', color='#d9534f')

    plt.ylabel('Next-Token Probability')
    plt.title(f'Phi-4 Next-Token Distribution Alignment\n(Sample {args.sample_idx}, Drop: {(1 - n_comp/n_orig)*100:.1f}%, KL: {kl:.4f})')
    plt.xticks(x, tokens, rotation=25, ha='right')
    plt.ylim(0, max(p_vals.max(), q_vals.max()) * 1.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    print(f"\nSaved distribution comparison plot to {args.output_plot}")

if __name__ == "__main__":
    main()
