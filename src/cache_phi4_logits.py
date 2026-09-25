import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Cache top-K next-token logits from Phi-4.")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-4")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Phi-4 on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    cached_data = {}
    with torch.no_grad():
        for idx, item in enumerate(tqdm(samples, desc="Caching Phi-4 distributions")):
            sample_id = item.get("id", str(idx))
            prompt_text = item["prompt"]

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_len
            ).to(device)

            outputs = model(**inputs)
            last_logits = outputs.logits[:, -1, :].float()
            log_probs = torch.log_softmax(last_logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, k=args.top_k, dim=-1)

            cached_data[sample_id] = {
                "prompt": prompt_text,
                "top_log_probs": top_log_probs.squeeze(0).cpu().to(torch.float16),
                "top_indices": top_indices.squeeze(0).cpu()
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(cached_data, args.output_path)
    print(f"Done. Saved {len(cached_data)} targets to {args.output_path}")

if __name__ == "__main__":
    main()
