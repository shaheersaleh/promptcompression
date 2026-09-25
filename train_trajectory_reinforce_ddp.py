import argparse
import json
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def init_distributed():
    """Initializes the NCCL distributed process group and sets local CUDA device."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

def format_chatml(context: str, question: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer questions based strictly on the context.<|im_end|>\n"
        f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\nAnswer:"
    )

class JsonlDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int = None):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def load_compressor_ddp(checkpoint_path: str, local_rank: int):
    """Loads RoBERTa policy model and wraps in PyTorch DDP."""
    device = f"cuda:{local_rank}"
    tokenizer = None
    candidates = [
        checkpoint_path,
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "xlm-roberta-large",
    ]
    for target in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
            break
        except Exception:
            continue

    if tokenizer is None:
        raise RuntimeError("Could not load compressor tokenizer offline.")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path,
        local_files_only=True
    ).to(device)
    model.train()

    # Wrap in DistributedDataParallel
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    return tokenizer, ddp_model

def load_evaluator_local(preferred_model: str, local_rank: int):
    """Loads 4-bit Phi-4 strictly onto the assigned local GPU (NO DDP wrapper)."""
    device = f"cuda:{local_rank}"
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
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            mod = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map={"": local_rank},  # Critical for DDP: isolates rank to GPU i
                local_files_only=True
            ).eval()
            tokenizer = tok
            model = mod
            selected_id = model_id
            break
        except Exception:
            continue

    if model is None or tokenizer is None:
        raise RuntimeError(f"Rank {local_rank}: No cached evaluator found offline.")

    return tokenizer, model, selected_id

def get_trajectory_logits(model, tokenizer, prompt_str: str, answer_ids: torch.Tensor, device: str) -> torch.Tensor:
    """Computes all T trajectory logits in a single causal forward pass."""
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(device)
    causal_input = torch.cat([prompt_ids, answer_ids[:, :-1]], dim=1)

    with torch.no_grad():
        outputs = model(causal_input)
        all_logits = outputs.logits

    prompt_len = prompt_ids.shape[1]
    target_logits = all_logits[0, prompt_len - 1 : prompt_len - 1 + answer_ids.shape[1], :]
    return target_logits

def compute_sparse_trajectory_kl(uncomp_logits: torch.Tensor, comp_logits: torch.Tensor, top_k: int = 100) -> float:
    probs_p = F.softmax(uncomp_logits, dim=-1)
    probs_q = F.softmax(comp_logits, dim=-1)

    topk_p, topk_idx = torch.topk(probs_p, top_k, dim=-1)
    topk_q = torch.gather(probs_q, dim=-1, index=topk_idx) + 1e-12

    step_kl = torch.sum(topk_p * torch.log(topk_p / topk_q), dim=-1)
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
    kept_tokens = [tok for tok, m in zip(context_tokens, mask) if m.item() == 1]
    comp_context = comp_tokenizer.convert_tokens_to_string(kept_tokens).strip()
    comp_prompt = format_chatml(comp_context, question)

    comp_logits = get_trajectory_logits(eval_model, eval_tokenizer, comp_prompt, answer_ids, device)
    mean_kl = compute_sparse_trajectory_kl(uncomp_logits, comp_logits, top_k=100)

    orig_len = len(mask)
    comp_len = max(1, int(mask.sum().item()))
    drop_rate = (orig_len - comp_len) / orig_len if orig_len > 0 else 0.0

    return -mean_kl + beta * drop_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/checkpoint_epoch_1")
    parser.add_argument("--eval_model", type=str, default="microsoft/phi-4")
    parser.add_argument("--output_dir", type=str, default="checkpoints/trajectory_rl_ddp")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_answer_tokens", type=int, default=24)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=250)
    args = parser.parse_args()

    # 1. Distributed Initialization
    local_rank, global_rank, world_size = init_distributed()
    device = f"cuda:{local_rank}"

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[*] Initialized DDP across {world_size} GPUs.", flush=True)

    # 2. Load Models
    comp_tokenizer, comp_ddp = load_compressor_ddp(args.base_checkpoint, local_rank)
    optimizer = torch.optim.AdamW(comp_ddp.parameters(), lr=args.lr)

    eval_tokenizer, eval_model, active_eval_id = load_evaluator_local(args.eval_model, local_rank)
    if global_rank == 0:
        print(f"[*] Active Evaluator: {active_eval_id}", flush=True)

    # 3. Load Sharded Dataset
    dataset = JsonlDataset(args.data_path, max_samples=args.max_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)

    if global_rank == 0:
        print(f"[*] Total dataset size: {len(dataset)} | Samples per GPU: {len(dataloader)}", flush=True)

    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        for step, batch_item in enumerate(dataloader):
            global_step += 1
            item = {k: v[0] if isinstance(v, list) else v for k, v in batch_item.items()}

            # Extract context and question
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
            logits = comp_ddp(**enc).logits[0, :, 1]
            keep_probs = torch.sigmoid(logits)

            policy_dist = Bernoulli(probs=keep_probs)
            sampled_mask = policy_dist.sample()
            greedy_mask = (keep_probs >= 0.5).float()

            # Target answer extraction
            answer_text = item.get("gold_answer") or item.get("answer")
            if answer_text:
                answer_ids = eval_tokenizer.encode(
                    " " + str(answer_text).strip(),
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(device)[:, :args.max_answer_tokens]
            else:
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
                    answer_ids = gen_out[:, uncomp_inp.input_ids.shape[1]:]

            if answer_ids.shape[1] == 0:
                continue

            uncomp_prompt = format_chatml(context, question)
            uncomp_logits = get_trajectory_logits(eval_model, eval_tokenizer, uncomp_prompt, answer_ids, device)

            # Evaluate rewards
            r_sampled = evaluate_mask(
                context_tokens, sampled_mask, question, answer_ids, uncomp_logits,
                eval_model, eval_tokenizer, comp_tokenizer, args.beta, device
            )
            r_greedy = evaluate_mask(
                context_tokens, greedy_mask, question, answer_ids, uncomp_logits,
                eval_model, eval_tokenizer, comp_tokenizer, args.beta, device
            )

            advantage = r_sampled - r_greedy

            # Policy gradient loss
            log_probs = policy_dist.log_prob(sampled_mask).sum()
            loss = -log_probs * advantage

            # Backpropagation (Synchronizes gradients across 4 GPUs automatically)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comp_ddp.parameters(), max_norm=1.0)
            optimizer.step()

            drop_pct = (1.0 - (sampled_mask.sum().item() / len(sampled_mask))) * 100

            # Logging on Rank 0
            if global_rank == 0 and global_step % args.log_interval == 0:
                print(
                    f"[GPU 0] Step {global_step:05d} | Loss: {loss.item():.4f} | "
                    f"Adv: {advantage:+.4f} | R_sample: {r_sampled:.3f} | R_base: {r_greedy:.3f} | "
                    f"Drop: {drop_pct:.1f}%",
                    flush=True
                )

            # Periodic checkpointing
            if global_step % args.save_interval == 0:
                dist.barrier()
                if global_rank == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                    comp_ddp.module.save_pretrained(save_path)
                    comp_tokenizer.save_pretrained(save_path)
                    print(f"[✓] Checkpoint saved at global step {global_step} -> {save_path}", flush=True)
                dist.barrier()

    # Final Save
    dist.barrier()
    if global_rank == 0:
        final_path = os.path.join(args.output_dir, "checkpoint_final")
        comp_ddp.module.save_pretrained(final_path)
        comp_tokenizer.save_pretrained(final_path)
        print(f"\n[✓] 4-GPU Training Complete. Final model saved to: {final_path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
