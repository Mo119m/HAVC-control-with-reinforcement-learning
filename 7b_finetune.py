# 7B + LoRA on Colab 40GB (bf16, no ref-KL). colab
import os, json, time, random, glob
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SYSTEM_MSG = "You are an HVAC controller. Reply ONLY with a JSON array of floats (values between -1 and 1), one line, nothing else."

LR = float(os.getenv("LR", "1e-5"))
EPOCHS = int(os.getenv("EPOCHS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))          # 40GB 建议设 1，配合梯度累积
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "1500"))
CLIP_EPS = float(os.getenv("CLIP_EPS", "0.2"))
ENTROPY_COEF = float(os.getenv("ENTROPY_COEF", "0.01"))
VALUE_COEF = float(os.getenv("VALUE_COEF", "1.0"))

# 不加载参考模型，省一份 7B 显存
KL_COEF = float(os.getenv("KL_COEF", "0.0"))            # 设为 0，完全关闭 KL

USE_LORA = (os.getenv("USE_LORA", "1") == "1") and PEFT_AVAILABLE
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 32, 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# 梯度累积（等效 batch = BATCH_SIZE * GRAD_ACCUM）
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))

SEED = int(os.getenv("SEED", "20240814"))
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

CANDIDATES = os.getenv(
    "ROLLOUT_GLOBS",
    "/content/drive/MyDrive/BEAR_LLM/outputs/mini_rollout_fewshot.json"
).split(",")

SAVE_DIR = os.getenv("SAVE_DIR", "/content/drive/MyDrive/BEAR_LLM/ft_out_offline_ppo_7b_lora")

REWARD_Q_LOW  = float(os.getenv("REWARD_Q_LOW",  "0.05"))
REWARD_Q_HIGH = float(os.getenv("REWARD_Q_HIGH", "0.99"))

ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")  # 可改 "eager"

def _is_clean_entry(e: Dict[str, Any]):
    if e.get("used_fallback"): return False
    if e.get("parsed_from") not in {"json","actions_line","any_brackets","last_json","forced_actions_line"}: return False
    au = e.get("action_unit")
    if not isinstance(au, list) or len(au)==0: return False
    try:
        if max(abs(float(x)) for x in au) > 1.05: return False
    except Exception:
        return False
    return True

def load_clean_rollouts(paths: List[str], enforce_file_boundary: bool=True):
    items = []
    for p in paths:
        data = json.load(open(p, "r", encoding="utf-8"))
        n_before = len(items)
        for e in data:
            if not _is_clean_entry(e):
                continue
            items.append({
                "prompt": e["prompt"],
                "answer": "[" + ", ".join(f"{float(a):.3f}" for a in e["action_unit"]) + "]",
                "reward": float(e.get("reward", 0.0)),
                "done": bool(e.get("done", False)),
            })
        if enforce_file_boundary and len(items) > n_before:
            items[-1]["done"] = True
    return items

class RolloutDataset(Dataset):
    def __init__(self, examples): self.data = examples
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x = self.data[i].copy()
        x["idx"] = i
        return x

def encode_one_sample(tok: AutoTokenizer, prompt_text: str, answer_text: str, max_len: int):
    msgs_full = [
        {"role":"system","content":SYSTEM_MSG},
        {"role":"user","content":prompt_text.strip()},
        {"role":"assistant","content":answer_text.strip()},
    ]
    ids_full = tok.apply_chat_template(msgs_full, tokenize=True, add_generation_prompt=False, return_tensors=None)
    msgs_prompt_only = [
        {"role":"system","content":SYSTEM_MSG},
        {"role":"user","content":prompt_text.strip()},
    ]
    ids_prompt_only = tok.apply_chat_template(msgs_prompt_only, tokenize=True, add_generation_prompt=True, return_tensors=None)
    if len(ids_full) > max_len: ids_full = ids_full[-max_len:]
    if len(ids_prompt_only) > max_len: ids_prompt_only = ids_prompt_only[-max_len:]
    labels = ids_full.copy()
    for i in range(min(len(ids_prompt_only), len(labels))): labels[i] = -100
    return ids_full, ids_prompt_only, labels

def collate_chat(batch, tok: AutoTokenizer, max_length=MAX_SEQ_LEN):
    enc_full, enc_prompt_only, enc_labels = [], [], []
    rewards, dones, idxs = [], [], []
    for ex in batch:
        ids_full, ids_prompt_only, labels = encode_one_sample(tok, ex["prompt"], ex["answer"], max_length)
        enc_full.append(torch.tensor(ids_full, dtype=torch.long))
        enc_prompt_only.append(torch.tensor(ids_prompt_only, dtype=torch.long))
        enc_labels.append(torch.tensor(labels, dtype=torch.long))
        rewards.append(float(ex["reward"])); dones.append(1.0 if ex["done"] else 0.0); idxs.append(ex["idx"])
    pad_id = tok.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(enc_full, batch_first=True, padding_value=pad_id)
    prompt_ids = torch.nn.utils.rnn.pad_sequence(enc_prompt_only, batch_first=True, padding_value=pad_id)
    labels    = torch.nn.utils.rnn.pad_sequence(enc_labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != pad_id).long()
    return (input_ids, attention_mask, labels,
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            prompt_ids, torch.tensor(idxs, dtype=torch.long))

class PPOModel(nn.Module):
    def __init__(self, base_lm: AutoModelForCausalLM, hidden_size: int):
        super().__init__()
        self.lm = base_lm
        lm_dtype = next(base_lm.parameters()).dtype
        self.value_head = nn.Linear(hidden_size, 1).to(dtype=lm_dtype)
    def forward(self, input_ids, attention_mask):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=True, return_dict=True)
        logits = out.logits
        h_last = out.hidden_states[-1]
        lengths = attention_mask.long().sum(dim=1) - 1
        b = torch.arange(logits.size(0), device=logits.device)
        values = self.value_head(h_last[b, lengths, :]).squeeze(-1)
        return logits, values

def shift_logits_and_labels(logits, labels, attention_mask):
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous(), attention_mask[:, 1:].contiguous()

def masked_token_stats(logits, labels, attention_mask, tok):
    logits, labels, attention_mask = shift_logits_and_labels(logits, labels, attention_mask)
    mask = (labels != -100) & (attention_mask != 0)
    if mask.sum() == 0:
        zeros = torch.zeros(logits.size(0), device=logits.device)
        return zeros, zeros, zeros
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = tok.pad_token_id
    lp_tok = log_probs.gather(2, labels_safe.unsqueeze(-1)).squeeze(-1)
    ent_tok = -(probs * log_probs).sum(dim=-1)
    B = logits.size(0)
    lp_avg, ent_avg, counts = [], [], []
    for i in range(B):
        m = mask[i]; cnt = m.sum().clamp_min(1); counts.append(cnt)
        lp_avg.append(lp_tok[i][m].mean()); ent_avg.append(ent_tok[i][m].mean())
    return torch.stack(lp_avg), torch.stack(ent_avg), torch.stack(counts).float()

def compute_gae(rews, vals, dones, gamma=0.99, lam=0.95):
    rews, vals, dones = rews.float(), vals.float(), dones.float()
    T = rews.size(0); adv = torch.zeros(T, device=rews.device, dtype=torch.float32); last = 0.0
    for t in reversed(range(T)):
        next_v = vals[t+1] if t < T-1 else 0.0
        mask = 1.0 - dones[t]
        delta = rews[t] + gamma * next_v * mask - vals[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv

# 主流程 
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 收集数据
    paths = []
    for pat in CANDIDATES: paths.extend(glob.glob(pat))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths: raise FileNotFoundError("找不到任何 rollout json，请先采集 mini_rollout_fewshot.json")
    data_all = load_clean_rollouts(paths, enforce_file_boundary=True)
    print(f"[Data] raw clean={len(data_all)}")

    # 奖励分位裁剪
    if len(data_all) >= 20 and (REWARD_Q_LOW > 0 or REWARD_Q_HIGH < 1):
        rews = torch.tensor([d["reward"] for d in data_all], dtype=torch.float32)
        lo = torch.quantile(rews, REWARD_Q_LOW).item()
        hi = torch.quantile(rews, REWARD_Q_HIGH).item()
        data_all = [d for d in data_all if (lo <= d["reward"] <= hi)]
        print(f"[Data] reward clip [{REWARD_Q_LOW:.2f},{REWARD_Q_HIGH:.2f}] -> {len(data_all)}")
    if len(data_all) == 0: raise RuntimeError("没有可用样本。")

    dataset = RolloutDataset(data_all)

    # 基座模型（bf16 + LoRA；不开 ref 模型；禁止用 flash-attn）
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    base = base.to(device)
    base.config.output_hidden_states = True
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False  # 训练期关闭 cache 省显存

    # 梯度检查点：显著省显存（会稍慢）
    try: base.gradient_checkpointing_enable()
    except Exception: pass

    if USE_LORA:
        if not PEFT_AVAILABLE: raise RuntimeError("PEFT 未安装，不能使用 LoRA。")
        lora_conf = LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES
        )
        base = get_peft_model(base, lora_conf)
        try: base.print_trainable_parameters()
        except Exception: pass

    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "n_embd", None)
    if hidden_size is None: raise ValueError("无法从 config 读取 hidden_size")

    policy_model = PPOModel(base, hidden_size).to(device)

    # 不加载 ref 模型（KL_COEF=0.0）
    ref_lm = None

    optimizer = AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=LR)

    # 预计算 old_logp / values（用当前 policy）
    with torch.no_grad():
        all_values, all_old_lp, all_rewards, all_dones = [], [], [], []
        value_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=lambda b: collate_chat(b, tok, MAX_SEQ_LEN))
        policy_model.eval()
        for input_ids, attn_mask, labels, rewards, dones, _, _ in value_loader:
            input_ids = input_ids.to(device); attn_mask = attn_mask.to(device); labels = labels.to(device)
            logits, values = policy_model(input_ids, attn_mask)
            old_lp, _, _ = masked_token_stats(logits, labels, attn_mask, tok)
            all_values.append(values.float().detach().cpu())
            all_old_lp.append(old_lp.float().detach().cpu())
            all_rewards.append(rewards); all_dones.append(dones)
        values_vec = torch.cat(all_values, dim=0)
        old_lp_vec = torch.cat(all_old_lp, dim=0)
        rewards_vec = torch.cat(all_rewards, dim=0)
        dones_vec   = torch.cat(all_dones, dim=0)

    def recompute_old_dists():
        nonlocal values_vec, old_lp_vec
        with torch.no_grad():
            all_values, all_old_lp = [], []
            policy_model.eval()
            for input_ids, attn_mask, labels, *_ in DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=lambda b: collate_chat(b, tok, MAX_SEQ_LEN)
            ):
                input_ids = input_ids.to(device); attn_mask = attn_mask.to(device); labels = labels.to(device)
                logits, values = policy_model(input_ids, attn_mask)
                old_lp, _, _ = masked_token_stats(logits, labels, attn_mask, tok)
                all_values.append(values.float().detach().cpu())
                all_old_lp.append(old_lp.float().detach().cpu())
            values_vec = torch.cat(all_values, dim=0)
            old_lp_vec = torch.cat(all_old_lp, dim=0)

    def build_adv_targets():
        adv = compute_gae(rewards_vec, values_vec, dones_vec, gamma=0.99, lam=0.95)
        m, s = adv.mean(), adv.std().clamp_min(1e-6)
        adv = ((adv - m) / s).clamp(-5.0, 5.0)
        return adv, (adv + values_vec)

    # idx 映射（按非打乱顺序）
    index_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: collate_chat(b, tok, MAX_SEQ_LEN))
    ordered_indices = []
    for *_, idxs in index_loader: ordered_indices.extend(idxs.tolist())
    idx_to_pos = {idx: pos for pos, idx in enumerate(ordered_indices)}

    advantages, value_targets = build_adv_targets()
    advantages = advantages.to(device); value_targets = value_targets.to(device)
    old_lp_vec = old_lp_vec.to(device)

    for ep in range(EPOCHS):
        t0 = time.time()

        # 每个 epoch 重算旧分布（无旧模型拷贝）
        recompute_old_dists()
        advantages, value_targets = build_adv_targets()
        advantages = advantages.to(device); value_targets = value_targets.to(device)
        old_lp_vec = old_lp_vec.to(device)

        policy_model.train()
        total_loss = total_pl = total_vl = total_el = total_kl = 0.0
        micro_steps = 0  # 统计累积步

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_chat(b, tok, MAX_SEQ_LEN))

        optimizer.zero_grad(set_to_none=True)

        for input_ids, attn_mask, labels, _, _, _, idxs in train_loader:
            input_ids = input_ids.to(device); attn_mask = attn_mask.to(device); labels = labels.to(device)
            idxs = idxs.to(device)
            pos = torch.tensor([idx_to_pos[i.item()] for i in idxs], device=device, dtype=torch.long)
            adv_b = advantages[pos]
            vt_b  = value_targets[pos]
            old_lp_b = old_lp_vec[pos]

            logits, values_pred = policy_model(input_ids, attn_mask)
            new_lp, new_ent, _ = masked_token_stats(logits, labels, attn_mask, tok)

            ratio = torch.exp(new_lp - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL(new || ref) 关闭
            kl = torch.tensor(0.0, device=device)

            value_loss = F.mse_loss(values_pred.float(), vt_b)
            entropy_loss = -ENTROPY_COEF * new_ent.mean()
            kl_loss = KL_COEF * kl

            loss = (policy_loss + VALUE_COEF * value_loss + entropy_loss + kl_loss) / GRAD_ACCUM
            loss.backward()
            micro_steps += 1

            if (micro_steps % GRAD_ACCUM) == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += (policy_loss + VALUE_COEF * value_loss + (-entropy_loss) + kl_loss).item()
            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            total_el += (-entropy_loss.item())
            total_kl += kl.item()

        denom = max(1, micro_steps // GRAD_ACCUM)
        print(f"[Epoch {ep+1}] {time.time()-t0:.1f}s | "
              f"Loss={total_loss/denom:.4f} | Policy={total_pl/denom:.4f} | "
              f"Value={total_vl/denom:.4f} | Entropy={total_el/denom:.4f} | KL={total_kl/denom:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    policy_model.lm.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    torch.save(policy_model.state_dict(), f"{SAVE_DIR}/policy_model.pt")
    print(f"[Saved] {SAVE_DIR}")

    # 验证
    policy_model.eval()
    idx = random.randint(0, len(dataset)-1)
    prompt_demo = dataset[idx]["prompt"]
    msgs = [{"role":"system","content":SYSTEM_MSG},{"role":"user","content":prompt_demo.strip()}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    enc = enc[:, -MAX_SEQ_LEN:].to(device)
    gen = policy_model.lm.generate(enc, max_new_tokens=40, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
    print("\n[Sample Prompt]\n", prompt_demo[:800])
    print("\n[Model Output]\n", tok.decode(gen[0], skip_special_tokens=True)[-400:])

if __name__ == "__main__":
    main()
