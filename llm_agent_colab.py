# llm_agent_colab.py — 允许推理；仅 user 消息；解析器更稳（Actions 行优先 & 无括号兜底）
import os, re, json
from typing import Optional, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEBUG = True
def dprint(*a, **k):
    if DEBUG: print(*a, **k)

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256     # 给“先推理 + 最后一行数组”留足空间
DEFAULT_TEMP = 0.7               # 允许推理
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.0  # 等效“frequency penalty = 0.0”

def _pick_device_map_and_dtype():
    """默认精度加载（无 4/8bit 开关）。"""
    if torch.cuda.is_available():
        try:
            return "auto", torch.bfloat16
        except Exception:
            return "auto", torch.float16
    return {"": "cpu"}, torch.float32

def _load_llm(model_name: str, hf_token: Optional[str] = None):
    dprint(f"[llm] loading: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    device_map, torch_dtype = _pick_device_map_and_dtype()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
        device_map=device_map,
        torch_dtype=torch_dtype
    ).eval()
    dprint("[llm] loaded ok")
    return tok, mdl

def call_llm(
    prompt: str,
    n_actions: int,
    model_name: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
):
    """
    允许模型先做简短推理；最后一行输出 Actions: [x1,...,xN]。
    这里只发送 user 消息；不再注入“只输出 JSON”的 system prompt。
    """
    global _TOKENIZER, _MODEL
    if _MODEL is None or _TOKENIZER is None:
        mn = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
        _TOKENIZER, _MODEL = _load_llm(mn, os.getenv("HF_TOKEN"))

    device = next(_MODEL.parameters()).device
    messages = [{"role": "user", "content": prompt}]

    inputs = _TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if DEBUG: dprint({"tokens_after_template": int(inputs.shape[1])})

    model_max = getattr(_MODEL.config, "max_position_embeddings", None) or 4096
    safe_cap = min(int(model_max), 8192)
    if inputs.shape[1] > safe_cap:
        inputs = inputs[:, -safe_cap:]

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        pad_token_id=_TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
        eos_token_id=[_TOKENIZER.eos_token_id],   
        use_cache=True,
        repetition_penalty=float(repetition_penalty),
    )
    if temperature and float(temperature) > 0.0:
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        ))
    else:
        gen_kwargs["do_sample"] = False

    out = _MODEL.generate(inputs.to(device), **gen_kwargs)
    gen = out[0, inputs.shape[1]:]
    text = _TOKENIZER.decode(gen, skip_special_tokens=True).strip()
    if DEBUG: dprint("[call_llm] gen tail:\n" + text[-200:])
    return text

# 优先匹配 “Actions: [ ... ]” 这一行；其次匹配无括号的 “Actions: 0.1, 0.2, ...”；
# 再兜底扫描任意一对中括号。
_ACT_LINE_RE = re.compile(r'Actions\s*:\s*(\[[^\[\]]*\])', re.IGNORECASE)
_ACT_LINE_NOBRACKETS_RE = re.compile(r'Actions\s*:\s*([\-0-9\.,\s]+)$', re.IGNORECASE)
_ANY_BRACKET_NUMS_RE = re.compile(r'\[\s*([^\[\]]*?)\s*\]')

def parse_actions(raw_text: str, n: int) -> Tuple[Optional[List[float]], dict]:
    """
    从整段文本中提取长度为 n 的数组（对“漏括号”的 Actions 行也做兜底）。
    优先级：
      1) 包含 'Actions: [ ... ]' 的行（从后往前找）
      2) 包含 'Actions: 0.1, 0.2, ...' 的行（无括号兜底）
      3) 最后一行尝试 JSON
      4) 全文任意一对中括号（最后一个长度匹配的）
    """
    text = re.sub(r"```[\w\-]*", "", raw_text).replace("```", "")
    text = text.replace("\u200b", "").replace("\u00a0", " ").strip()
    meta = {"parsed_from": "failed"}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    #优先：带括号的 Actions 行
    for ln in reversed(lines):
        m = _ACT_LINE_RE.search(ln)
        if m:
            try:
                arr = json.loads(m.group(1))
                if isinstance(arr, list) and len(arr) == n:
                    meta["parsed_from"] = "actions_line"
                    return [float(x) for x in arr], meta
            except Exception:
                pass

    # 兜底：不带括号的 Actions 行（用逗号切分）
    for ln in reversed(lines):
        m = _ACT_LINE_NOBRACKETS_RE.search(ln)
        if m:
            try:
                parts = [p.strip() for p in m.group(1).split(",") if p.strip() != ""]
                arr = [float(x) for x in parts]
                if len(arr) == n:
                    meta["parsed_from"] = "actions_line_no_brackets"
                    return arr, meta
            except Exception:
                pass

    # 次选：最后一行整段尝试 JSON（与原逻辑保持）
    try:
        last = lines[-1] if lines else ""
        if last.count("[") >= 1 and last.count("]") == 0:
            last = last + "]"
        data = json.loads(last)
        if isinstance(data, list) and len(data) == n:
            meta["parsed_from"] = "last_json"
            return [float(x) for x in data], meta
    except Exception:
        pass

    # 兜底：全文扫描任意中括号（取最后一个长度匹配的）
    last_good = None
    for m in _ANY_BRACKET_NUMS_RE.finditer(text):
        inner = m.group(1)
        try:
            arr = [float(x.strip()) for x in inner.split(",") if x.strip() != ""]
        except Exception:
            continue
        if len(arr) == n:
            last_good = arr

    if last_good is not None:
        meta["parsed_from"] = "any_brackets"
        return last_good, meta

    return None, meta
