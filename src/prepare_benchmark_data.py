import argparse
import json
import os
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare benchmark dataset with ChatML format.")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--output_path", type=str, default="data/train.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print("Loading SQuAD train split...")
    ds = load_dataset("rajpurkar/squad", split="train")

    formatted = []
    for i, row in enumerate(ds):
        if len(formatted) >= args.num_samples:
            break
        context = row["context"].strip()
        question = row["question"].strip()
        
        # Filter for moderately long contexts (between 80 and 300 words)
        words = context.split()
        if len(words) < 80 or len(words) > 300:
            continue

        # Format with ChatML template so Phi-4 expects generation
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant. Answer questions based strictly on the context.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\nAnswer:"
        )
        formatted.append({"id": f"squad_{i:04d}", "prompt": prompt})

    with open(args.output_path, "w", encoding="utf-8") as f:
        for entry in formatted:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully wrote {len(formatted)} ChatML prompts to {args.output_path}")

if __name__ == "__main__":
    main()
