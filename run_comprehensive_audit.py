import argparse
import json
import os
import string
from collections import Counter
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def compute_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def compute_em(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def format_chatml(context: str, question: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer questions based strictly on the context.<|im_end|>\n"
        f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\nAnswer:"
    )

def load_compressor_tokenizer(checkpoint_path: str):
    candidates = [
        checkpoint_path,
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "xlm-roberta-large",
    ]
    for target in candidates:
        try:
            print(f"[*] Attempting to load compressor tokenizer from: {target}")
            tok = AutoTokenizer.from_pretrained(target, local_files_only=True)
            print(f"[✓] Successfully loaded compressor tokenizer from: {target}")
            return tok
        except Exception as e:
            print(f"[-] Could not load from {target}: {e}")
            continue
    raise RuntimeError("Failed to resolve a working tokenizer for the compressor offline.")

def load_evaluator_pipeline(preferred_model: str, device: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    candidates = [preferred_model, "microsoft/phi-4", "microsoft/Phi-4-mini-instruct"]
    seen = set()
    ordered_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    tokenizer = None
    model = None
    selected_id = None

    for model_id in ordered_candidates:
        try:
            print(f"[*] Attempting to load offline evaluator: {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            mod = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True
            ).eval()
            tokenizer = tok
            model = mod
            selected_id = model_id
            print(f"[✓] Evaluator loaded successfully: {selected_id}")
            break
        except Exception as e:
            print(f"[-] Could not load {model_id} offline: {e}")
            continue

    if model is None or tokenizer is None:
        raise RuntimeError("No cached evaluator model found offline. Verify ~/.cache or $HF_HOME.")

    return tokenizer, model, selected_id

def compress_context(
    context: str,
    tokenizer,
    model,
    device: str,
    threshold: float = 0.5
) -> Tuple[str, int, int, float]:
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0, :, 1]  # P(keep) at index 1

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    kept_tokens = [
        tok for tok, p in zip(tokens, probs)
        if p >= threshold and tok not in ("<s>", "</s>", "<pad>")
    ]
    compressed_text = tokenizer.convert_tokens_to_string(kept_tokens).strip()

    orig_len = len(inputs["input_ids"][0])
    comp_len = len(kept_tokens)
    drop_rate = (orig_len - comp_len) / orig_len if orig_len > 0 else 0.0
    return compressed_text, orig_len, comp_len, drop_rate

def generate_answer(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 48
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>")
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

def audit_trajectory_probabilities(
    uncomp_prompt: str,
    comp_prompt: str,
    target_answer: str,
    model,
    tokenizer,
    device: str,
    max_audit_tokens: int = 15
) -> Tuple[List[Dict], float, float, float]:
    answer_tokens = tokenizer.encode(" " + target_answer.strip(), add_special_tokens=False)[:max_audit_tokens]
    if not answer_tokens:
        return [], 1.0, 1.0, 1.0

    uncomp_prefix_ids = tokenizer.encode(uncomp_prompt, add_special_tokens=False)
    comp_prefix_ids = tokenizer.encode(comp_prompt, add_special_tokens=False)

    step_records = []
    cumul_p = 1.0
    cumul_q = 1.0

    curr_uncomp_ids = list(uncomp_prefix_ids)
    curr_comp_ids = list(comp_prefix_ids)

    with torch.no_grad():
        for step_idx, tok_id in enumerate(answer_tokens):
            tok_str = tokenizer.decode([tok_id])

            # Forward pass uncompressed
            inp_p = torch.tensor([curr_uncomp_ids], device=device)
            logits_p = model(inp_p).logits[0, -1, :]
            probs_p = F.softmax(logits_p, dim=-1)
            p_val = float(probs_p[tok_id].item())

            # Forward pass compressed
            inp_q = torch.tensor([curr_comp_ids], device=device)
            logits_q = model(inp_q).logits[0, -1, :]
            probs_q = F.softmax(logits_q, dim=-1)
            q_val = float(probs_q[tok_id].item())

            # Rank of target token under Q
            rank_q = int((probs_q > probs_q[tok_id]).sum().item()) + 1

            # Top-100 Sparse KL divergence at this step
            top100_vals, top100_idx = torch.topk(probs_p, 100)
            p_sparse = top100_vals
            q_sparse = probs_q[top100_idx] + 1e-12
            step_kl = float(torch.sum(p_sparse * torch.log(p_sparse / q_sparse)).item())

            cumul_p *= p_val
            cumul_q *= q_val

            step_records.append({
                "step": step_idx + 1,
                "token": tok_str,
                "token_id": tok_id,
                "p_t": round(p_val, 4),
                "cumul_p": f"{cumul_p:.4e}",
                "q_t": round(q_val, 4),
                "cumul_q": f"{cumul_q:.4e}",
                "rank_q": rank_q,
                "step_kl": round(max(0.0, step_kl), 4)
            })

            curr_uncomp_ids.append(tok_id)
            curr_comp_ids.append(tok_id)

    likelihood_ratio = cumul_q / cumul_p if cumul_p > 0 else 0.0
    return step_records, cumul_p, cumul_q, likelihood_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--output_file", type=str, default="comprehensive_audit_results.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_1")
    parser.add_argument("--eval_model", type=str, default="microsoft/phi-4")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing evaluation pipeline on device: {device}")

    # 1. Load RoBERTa compressor tokenizer and model weights
    comp_tokenizer = load_compressor_tokenizer(args.checkpoint)
    print(f"[*] Loading compressor model from checkpoint: {args.checkpoint}")
    comp_model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        local_files_only=True
    ).to(device).eval()

    # 2. Load Downstream Evaluator with offline fallback
    phi4_tokenizer, phi4_model, active_eval_id = load_evaluator_pipeline(args.eval_model, device)
    print(f"[*] Active evaluator model: {active_eval_id}")

    # 3. Read existing output file to resume seamlessly if interrupted
    completed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["sample_id"])
                    except json.JSONDecodeError:
                        continue
        if completed_ids:
            print(f"[*] Resuming '{args.output_file}': {len(completed_ids)} samples already complete.")

    # 4. Load dataset
    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= args.num_samples:
                break

    print(f"[*] Target dataset contains {len(samples)} samples to evaluate.")

    # 5. Autoregressive evaluation and real-time disk sync loop
    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(samples):
            sample_id = idx + 1
            if sample_id in completed_ids:
                continue

            prompt_str = item.get("prompt", "")
            if "Context: " in prompt_str and "\n\nQuestion:" in prompt_str:
                _, rest = prompt_str.split("Context: ", 1)
                context, question_part = rest.split("\n\nQuestion:", 1)
                question = question_part.replace("<|im_end|>", "").replace("<|im_start|>assistant\nAnswer:", "").strip()
            else:
                context = item.get("context", "")
                question = item.get("question", "")

            gold_reference = item.get("gold_answer", item.get("answer", ""))

            # Baseline uncompressed generation
            uncomp_prompt = format_chatml(context, question)
            uncomp_ans = generate_answer(uncomp_prompt, phi4_model, phi4_tokenizer, device)

            # Context compression
            comp_context, orig_toks, comp_toks, drop_rate = compress_context(
                context, comp_tokenizer, comp_model, device, threshold=args.threshold
            )
            comp_prompt = format_chatml(comp_context, question)
            comp_ans = generate_answer(comp_prompt, phi4_model, phi4_tokenizer, device)

            # Autoregressive trajectory probability trace
            steps_data, cumul_p, cumul_q, l_ratio = audit_trajectory_probabilities(
                uncomp_prompt, comp_prompt, uncomp_ans, phi4_model, phi4_tokenizer, device
            )

            # Agreement metrics
            f1 = compute_f1(comp_ans, uncomp_ans)
            em = compute_em(comp_ans, uncomp_ans)

            record = {
                "sample_id": sample_id,
                "question": question,
                "gold_reference": gold_reference,
                "uncompressed_context": context,
                "uncompressed_prompt": uncomp_prompt,
                "uncompressed_tokens": orig_toks,
                "uncompressed_answer": uncomp_ans,
                "compressed_context": comp_context,
                "compressed_prompt": comp_prompt,
                "compressed_tokens": comp_toks,
                "token_drop_rate_pct": round(drop_rate * 100, 2),
                "compressed_answer": comp_ans,
                "metrics": {
                    "token_f1": round(f1, 4),
                    "exact_match": em,
                    "likelihood_ratio_q_over_p": round(l_ratio, 4),
                    "cumul_p": f"{cumul_p:.4e}",
                    "cumul_q": f"{cumul_q:.4e}"
                },
                "probability_audit_steps": steps_data
            }

            # Force immediate flush to disk
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            print(
                f"[✓] Processed Sample {sample_id:02d}/{len(samples)} | "
                f"Drop: {record['token_drop_rate_pct']}% | "
                f"F1: {f1:.2f} | EM: {em} | Q/P Ratio: {l_ratio:.4f}",
                flush=True
            )

    print(f"\n[✓] Audit successfully completed. All output committed to {args.output_file}")

if __name__ == "__main__":
    main()
