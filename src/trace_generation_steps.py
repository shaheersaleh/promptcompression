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
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi4_id", type=str, default="microsoft/phi-4")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/checkpoint_epoch_1")
    parser.add_argument("--test_data", type=str, default="data/train.jsonl")
    parser.add_argument("--sample_idx", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=15)
    return parser.parse_args()

def generate_with_step_trace(model, tokenizer, prompt_text, max_tokens=15, device="cuda"):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    curr_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    trace = []
    generated_ids = []
    past_key_values = None

    with torch.no_grad():
        for step in range(max_tokens):
            if past_key_values is None:
                outputs = model(input_ids=curr_ids, attention_mask=attention_mask, use_cache=True)
            else:
                outputs = model(input_ids=curr_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            probs = F.softmax(logits, dim=-1).squeeze(0)

            next_token_id = torch.argmax(probs).item()
            next_prob = probs[next_token_id].item()

            token_str = tokenizer.decode([next_token_id])
            trace.append({
                "step": step + 1,
                "chosen_token": token_str,
                "confidence": next_prob,
            })
            generated_ids.append(next_token_id)

            if token_str in ["<|im_end|>", "\n", "<|endoftext|>"]:
                break

            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)

    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return full_output, trace

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

    print("=" * 80)
    print(f"TRACING SAMPLE {args.sample_idx}")
    print("=" * 80)
    print(f"[Original Context Tokens]:   {len(phi_tok(context_str)['input_ids'])}")
    print(f"[Compressed Context Tokens]: {len(phi_tok(comp_context)['input_ids'])}")
    print("-" * 80)

    orig_ans, orig_trace = generate_with_step_trace(phi_model, phi_tok, prompt, max_tokens=args.max_tokens, device=device)
    comp_ans, comp_trace = generate_with_step_trace(phi_model, phi_tok, comp_prompt, max_tokens=args.max_tokens, device=device)

    print(f"\n[Uncompressed Generation]: \"{orig_ans}\"")
    print(f"[Compressed Generation]:   \"{comp_ans}\"\n")

    print(f"{'Step':<5} | {'Orig Token (Conf)':<28} | {'Comp Token (Conf)':<28} | {'Match?'}")
    print("-" * 75)

    max_steps = max(len(orig_trace), len(comp_trace))
    for s in range(max_steps):
        t_orig = orig_trace[s] if s < len(orig_trace) else None
        t_comp = comp_trace[s] if s < len(comp_trace) else None

        o_str = f"'{t_orig['chosen_token']}' ({t_orig['confidence']:.2%})" if t_orig else "-"
        c_str = f"'{t_comp['chosen_token']}' ({t_comp['confidence']:.2%})" if t_comp else "-"

        match = "✓" if (t_orig and t_comp and t_orig["chosen_token"] == t_comp["chosen_token"]) else "≠"
        print(f"{s+1:<5} | {o_str:<28} | {c_str:<28} | {match}")

    print("-" * 75)

if __name__ == "__main__":
    main()
