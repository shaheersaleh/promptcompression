import json
import os
from datasets import load_dataset

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

output_path = "data/train_full.jsonl"
os.makedirs("data", exist_ok=True)

print("[*] Loading cached SQuAD dataset from ~/.cache/huggingface/datasets...")
dataset = load_dataset("squad", split="train", cache_dir="/home/shaheer/.cache/huggingface/datasets")
print(f"[*] Raw dataset loaded: {len(dataset)} samples")

written = 0
with open(output_path, "w", encoding="utf-8") as f:
    for idx, item in enumerate(dataset):
        context = item["context"].strip()
        question = item["question"].strip()
        
        # Extract first ground-truth answer for causal trajectory slicing
        answers = item.get("answers", {}).get("text", [])
        gold_answer = answers[0].strip() if answers else ""
        
        if not context or not question or not gold_answer:
            continue
            
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant. Answer questions based strictly on the context.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
            "<|im_start|>assistant\nAnswer:"
        )
        
        record = {
            "id": f"squad_{idx:05d}",
            "context": context,
            "question": question,
            "answer": gold_answer,
            "prompt": prompt
        }
        f.write(json.dumps(record) + "\n")
        written += 1

print(f"[✓] Successfully generated {output_path} with {written} samples.")
print(f"[*] File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
