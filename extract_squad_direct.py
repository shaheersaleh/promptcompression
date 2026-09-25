import json
import os
from datasets import Dataset

arrow_path = "/home/shaheer/.cache/huggingface/datasets/rajpurkar___squad/plain_text/0.0.0/7b6d24c440a36b6815f21b70d25016731768db1f/squad-train.arrow"
output_path = "data/train_full.jsonl"
os.makedirs("data", exist_ok=True)

print(f"[*] Reading directly from: {arrow_path}")
dataset = Dataset.from_file(arrow_path)
print(f"[*] Total dataset samples: {len(dataset)}")

written = 0
with open(output_path, "w", encoding="utf-8") as f:
    for idx, item in enumerate(dataset):
        context = item.get("context", "").strip()
        question = item.get("question", "").strip()
        
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

print(f"[✓] Extracted {written} samples to {output_path}")
print(f"[*] Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
