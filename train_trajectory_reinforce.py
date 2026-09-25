import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def format_chatml(context: str, question: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer questions based strictly on the context.<|im_end|>\n"
        f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\nAnswer:"
    )

def load_compressor_components(checkpoint_path: str, device: str):
    """Loads RoBERTa tokenizer and token-classification model with offline fallbacks."""
    print(f"[*] Loading compressor from: {checkpoint_path}")
    tokenizer = None
    candidates = [
        checkpoint_path,
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "xlm-roberta-large",
    ]
    for target in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
            print(f"[✓] Compressor tokenizer loaded from: {target}")
            break
        except Exception:
            continue

    if tokenizer is None:
        raise RuntimeError("Could not load compressor tokenizer from local checkpoint or cache.")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path,
        local_files_only=True
    ).to(device)
    model.train()
    return tokenizer, model

def load_evaluator_components(preferred_model: str, device: str):
    """Loads evaluator in 4-bit NF4 with local fallback to mini if 14B is missing."""
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
        raise RuntimeError("No cached evaluator model found. Ensure weights are cached on disk.")

    return tokenizer, model, selected_id

def get_trajectory_logits(model, tokenizer, prompt_str: str, answer_ids: torch.Tensor, device: str) -> torch.Tensor:
    """Computes downstream next-token logits for all T answer steps in a single forward pass."""
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(device)
    
    # Input: [Prompt Tokens] + [y_1, y_2, ..., y_{T-1}]
    causal_input = torch.cat([prompt_ids, answer_ids[:, :-1]], dim=1)
    
    with torch.no_grad():
        outputs = model(causal_input)
        all_logits = outputs.logits  # [1, Seq_Len, Vocab]
        
    prompt_len = prompt_ids.shape[1]
    # Sliced logits corresponding to target tokens [y_1, ..., y_T]
    target_logits = all_logits[0, prompt_len - 1 : prompt_len - 1 + answer_ids.shape[1], :]
    return target_logits  # [T, Vocab]

def compute_sparse_trajectory_kl(uncomp_logits: torch.Tensor, comp_logits: torch.Tensor, top_k: int = 100) -> float:
    """Calculates mean Top-K Sparse KL divergence across all T generation steps."""
    probs_p = F.softmax(uncomp_logits, dim=-1)  # [T, Vocab]
    probs_q = F.softmax(comp_logits, dim=-1)    # [T, Vocab]
    
    topk_p, topk_idx = torch.topk(probs_p, top_k, dim=-1)  # [T, K]
    topk_q = torch.gather(probs_q, dim=-1, index=topk_idx) + 1e-12
    
    step_kl = torch.sum(topk_p * torch.log(topk_p / topk_q), dim=-1)  # [T]
    return float(step_kl.mean().item())

def evaluate_mask(
    context_tokens: list,
    mask: torch.Tensor,
    question: str,
    answer_ids: torch.Tensor,
    uncomp_logits: torch.Tensor,
    eval_model,
    eval_tokenizer,
    comp_tokenizer,
    beta: float,
    device: str
) -> float:
    """Builds compressed context from binary mask, queries evaluator, and computes R(m)."""
    kept_tokens = [tok for tok, m in zip(context_tokens, mask) if m.item() == 1]
    comp_context = comp_tokenizer.convert_tokens_to_string(kept_tokens).strip()
    comp_prompt = format_chatml(comp_context, question)
    
    comp_logits = get_trajectory_logits(eval_model, eval_tokenizer, comp_prompt, answer_ids, device)
    mean_kl = compute_sparse_trajectory_kl(uncomp_logits, comp_logits, top_k=100)
    
    orig_len = len(mask)
    comp_len = max(1, int(mask.sum().item()))
    drop_rate = (orig_len - comp_len) / orig_len if orig_len > 0 else 0.0
    
    reward = -mean_kl + beta * drop_rate
    return reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/checkpoint_epoch_1")
    parser.add_argument("--eval_model", type=str, default="microsoft/phi-4")
    parser.add_argument("--output_dir", type=str, default="checkpoints/trajectory_rl")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_answer_tokens", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting Trajectory REINFORCE Training on: {device}")

    # 1. Load RoBERTa Policy Network
    comp_tokenizer, comp_model = load_compressor_components(args.base_checkpoint, device)
    optimizer = torch.optim.AdamW(comp_model.parameters(), lr=args.lr)

    # 2. Load Frozen Downstream Evaluator (with automatic fallback)
    eval_tokenizer, eval_model, active_eval_id = load_evaluator_components(args.eval_model, device)
    print(f"[*] Active evaluator for trajectory rewards: {active_eval_id}")

    # 3. Load Dataset
    samples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= args.max_samples:
                break
    print(f"[*] Loaded {len(samples)} samples for policy optimization.")

    step = 0
    for epoch in range(args.max_epochs):
        for item in samples:
            step += 1
            
            prompt_str = item.get("prompt", "")
            if "Context: " in prompt_str and "\n\nQuestion:" in prompt_str:
                _, rest = prompt_str.split("Context: ", 1)
                context, question_part = rest.split("\n\nQuestion:", 1)
                question = question_part.replace("<|im_end|>", "").replace("<|im_start|>assistant\nAnswer:", "").strip()
            else:
                context = item.get("context", "")
                question = item.get("question", "")

            enc = comp_tokenizer(context, return_tensors="pt", truncation=True, max_length=512).to(device)
            input_ids = enc["input_ids"][0]
            context_tokens = comp_tokenizer.convert_ids_to_tokens(input_ids)
            
            if len(context_tokens) < 10:
                continue

            # Forward pass through policy
            logits = comp_model(**enc).logits[0, :, 1]
            keep_probs = torch.sigmoid(logits)

            # Sample action mask and greedy baseline mask
            policy_dist = Bernoulli(probs=keep_probs)
            sampled_mask = policy_dist.sample()
            greedy_mask = (keep_probs >= 0.5).float()

            # Generate reference teacher answer trajectory from uncompressed context
            uncomp_prompt = format_chatml(context, question)
            with torch.no_grad():
                uncomp_inp = eval_tokenizer(uncomp_prompt, return_tensors="pt").to(device)
                gen_out = eval_model.generate(
                    **uncomp_inp,
                    max_new_tokens=args.max_answer_tokens,
                    do_sample=False,
                    pad_token_id=eval_tokenizer.eos_token_id,
                    eos_token_id=eval_tokenizer.encode("<|im_end|>")
                )
                answer_ids = gen_out[:, uncomp_inp.input_ids.shape[1]:]  # [1, T]

            if answer_ids.shape[1] == 0:
                continue

            # Reference uncompressed trajectory logits
            uncomp_logits = get_trajectory_logits(eval_model, eval_tokenizer, uncomp_prompt, answer_ids, device)

            # Rewards
            r_sampled = evaluate_mask(
                context_tokens, sampled_mask, question, answer_ids, uncomp_logits,
                eval_model, eval_tokenizer, comp_tokenizer, args.beta, device
            )
            r_greedy = evaluate_mask(
                context_tokens, greedy_mask, question, answer_ids, uncomp_logits,
                eval_model, eval_tokenizer, comp_tokenizer, args.beta, device
            )

            # SCST Advantage
            advantage = r_sampled - r_greedy

            # Policy Gradient Loss: - sum_i log pi(m_i) * Advantage
            log_probs = policy_dist.log_prob(sampled_mask).sum()
            loss = -log_probs * advantage

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comp_model.parameters(), max_norm=1.0)
            optimizer.step()

            drop_pct = (1.0 - (sampled_mask.sum().item() / len(sampled_mask))) * 100

            if step % 5 == 0:
                print(
                    f"Step {step:04d} | Loss: {loss.item():.4f} | Adv: {advantage:+.4f} | "
                    f"R_sample: {r_sampled:.3f} | R_base: {r_greedy:.3f} | Drop: {drop_pct:.1f}%",
                    flush=True
                )

            if step % 100 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_step_{step}")
                comp_model.save_pretrained(save_path)
                comp_tokenizer.save_pretrained(save_path)
                print(f"[✓] Checkpoint saved at step {step} -> {save_path}", flush=True)

    final_path = os.path.join(args.output_dir, "checkpoint_final")
    comp_model.save_pretrained(final_path)
    comp_tokenizer.save_pretrained(final_path)
    print(f"\n[✓] Training complete. Final model saved to: {final_path}", flush=True)

if __name__ == "__main__":
    main()
