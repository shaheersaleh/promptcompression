import argparse
import os
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Task-aware context compression RL.")
    parser.add_argument("--phi4_id", type=str, default="microsoft/phi-4")
    parser.add_argument("--encoder_id", type=str, default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    parser.add_argument("--cached_targets", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=2.5, help="Reward weight for compression rate")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=1536)
    return parser.parse_args()

def sparse_topk_kl(p_top_log_probs, p_top_indices, q_logits):
    p_probs = torch.exp(p_top_log_probs)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    q_selected_log_probs = torch.gather(q_log_probs, dim=-1, index=p_top_indices.unsqueeze(0)).squeeze(0)
    return torch.sum(p_probs * (p_top_log_probs - q_selected_log_probs))

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phi_tok = AutoTokenizer.from_pretrained(args.phi4_id, trust_remote_code=True)
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_id)

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
    for param in phi_model.parameters():
        param.requires_grad = False

    compressor = AutoModelForTokenClassification.from_pretrained(args.encoder_id).to(device)
    compressor.train()
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=args.lr)

    cached_targets = torch.load(args.cached_targets)
    sample_ids = list(cached_targets.keys())

    baseline = 0.0
    alpha = 0.95
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for step, sid in enumerate(sample_ids):
            target = cached_targets[sid]
            prompt = target["prompt"]
            p_top_log_probs = target["top_log_probs"].to(device).float()
            p_top_indices = target["top_indices"].to(device)

            # Split prompt: isolate context so instructions/questions remain untouched
            if "Context: " in prompt and "\n\nQuestion:" in prompt:
                prefix, rest = prompt.split("Context: ", 1)
                context_str, suffix = rest.split("\n\nQuestion:", 1)
                prefix += "Context: "
                suffix = "\n\nQuestion:" + suffix
            else:
                prefix, context_str, suffix = "", prompt, ""

            # 1. Forward context through compressor policy
            enc_inputs = enc_tok(context_str, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            enc_logits = compressor(**enc_inputs).logits.squeeze(0)
            keep_probs = torch.softmax(enc_logits, dim=-1)[:, 1]

            # 2. Sample keep/drop decisions
            dist = Bernoulli(probs=keep_probs)
            actions = dist.sample()
            if actions.sum() == 0:
                actions[torch.argmax(keep_probs)] = 1.0

            log_pi = dist.log_prob(actions).sum()
            entropy = dist.entropy().sum()

            # 3. Assemble compressed prompt
            input_tokens = enc_tok.convert_ids_to_tokens(enc_inputs["input_ids"].squeeze(0))
            kept_tokens = [tok for tok, act in zip(input_tokens, actions) if act.item() == 1.0]
            compressed_context = enc_tok.convert_tokens_to_string(kept_tokens).strip() or context_str
            full_compressed_prompt = prefix + compressed_context + suffix

            # 4. Forward compressed prompt through Phi-4
            phi_inputs = phi_tok(full_compressed_prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            with torch.no_grad():
                phi_outputs = phi_model(**phi_inputs)
                q_last_logits = phi_outputs.logits[:, -1, :].float()

            # 5. Reward computation: -KL + beta * drop_rate
            kl = sparse_topk_kl(p_top_log_probs, p_top_indices, q_last_logits)
            drop_rate = 1.0 - actions.mean()
            reward = -kl.item() + (args.beta * drop_rate.item())

            # 6. Policy gradient step with gradient accumulation
            advantage = reward - baseline
            baseline = alpha * baseline + (1.0 - alpha) * reward
            loss = (-log_pi * advantage - args.entropy_coef * entropy) / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(sample_ids):
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            del enc_inputs, enc_logits, dist, actions, phi_inputs, phi_outputs, q_last_logits
            torch.cuda.empty_cache()

            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(sample_ids)} | "
                      f"R: {reward:.4f} | KL: {kl.item():.4f} | Drop: {drop_rate.item():.2f}")

        save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        compressor.save_pretrained(save_path)
        enc_tok.save_pretrained(save_path)

if __name__ == "__main__":
    main()
