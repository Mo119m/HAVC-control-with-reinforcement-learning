# few_shot_auto.py — 自动选择高 reward + 最相似 few-shot，生成无中括号示例并注入 prompt
import json, math
from typing import List, Dict, Optional, Tuple

def zone_count_from_obs(obs: List[float]) -> int:
    try:
        n = (len(obs) - 2) // 3
        return n if n > 0 else 1
    except Exception:
        return 1

def _extract_env_terms(obs: List[float], n: int) -> Dict[str, float]:
    outside = float(obs[n]) if len(obs) > n else 0.0
    ghi_vals = [float(x) for x in obs[n+1:2*n+1]] if len(obs) >= 2*n+1 else []
    ghi_avg = (sum(ghi_vals) / len(ghi_vals)) if ghi_vals else 0.0
    ground_idx = 2*n + 1
    ground = float(obs[ground_idx]) if len(obs) > ground_idx else 0.0
    occ_vals = [float(x) for x in obs[2*n+2:3*n+2]] if len(obs) >= 3*n+2 else []
    occ_sum_kw = sum(occ_vals) if occ_vals else 0.0
    return {"outside": outside, "ghi_avg": ghi_avg, "ground": ground, "occ_sum_kw": occ_sum_kw}

def _featurize(obs: List[float], n_override: Optional[int] = None) -> List[float]:
    n = n_override if n_override is not None else zone_count_from_obs(obs)
    temps = [float(x) for x in (obs[:n] if len(obs) >= n else obs)]
    env = _extract_env_terms(obs, n)
    return temps + [env["outside"], env["ghi_avg"], env["ground"], env["occ_sum_kw"]]

def _euclid(a: List[float], b: List[float], w: Optional[List[float]] = None) -> float:
    L = min(len(a), len(b))
    if w is None: w = [1.0] * L
    s = 0.0
    for i in range(L):
        d = (a[i] - b[i])
        wi = w[i] if i < len(w) else 1.0
        s += wi * d * d
    return math.sqrt(s)

def load_examples(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def select_examples(
    dataset: List[Dict],
    current_obs: List[float],
    k: int = 3,
    alpha: float = 0.6,
    weights: Optional[List[float]] = None,
    building: Optional[str] = None,
    climate: Optional[str] = None,
    location: Optional[str] = None,
):
    n = zone_count_from_obs(current_obs)
    cur_feat = _featurize(current_obs, n_override=n)
    if weights is None:
        weights = [1.0]*n + [0.5, 0.2, 0.2, 0.1]

    def _ok(e: Dict) -> bool:
        if building and e.get("building") != building: return False
        if climate and e.get("climate") != climate: return False
        if location and e.get("location") != location: return False
        return True

    pool = [e for e in dataset if _ok(e)] or dataset
    rewards = [float(e.get("reward", 0.0)) for e in pool]
    rmin, rmax = min(rewards), max(rewards)
    rspan = (rmax - rmin) or 1.0

    scored: List[Tuple[float, Dict]] = []
    for e in pool:
        feat_e = _featurize(e.get("obs", []), n_override=n)
        dist = _euclid(cur_feat, feat_e, w=weights)
        sim = 1.0 / (1.0 + dist)
        r_norm = (float(e.get("reward", 0.0)) - rmin) / rspan
        score = alpha * sim + (1.0 - alpha) * r_norm
        scored.append((score, e))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:k]]

def format_few_shot_block(examples: List[Dict], target: float, n: int) -> str:
    lines = []
    for e in examples:
        obs = e.get("obs", [])
        rooms = [f"{float(x):.1f}" for x in obs[:n]]
        acts  = [f"{float(a):.1f}" for a in (e.get("actions", [])[:n] or [0.0]*n)]
        lines.append(
            f"- Example (reward={float(e.get('reward',0.0)):.3g}): "
            f"Rooms: {', '.join(rooms)}; Target: {float(target):.1f} -> "
            f"Actions: {', '.join(acts)}"
        )
    return "Auto-selected examples (similar & high-reward):\n" + "\n".join(lines)

def inject_few_shot(prompt: str, fewshot_block: str) -> str:
    anchor = "History Action And Feedback Reference:"
    idx = prompt.find(anchor)
    if idx == -1:
        return prompt.rstrip() + "\n\n" + fewshot_block.strip() + "\n"
    lines = prompt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(anchor):
            insert_at = i + 2  
            break
    else:
        insert_at = len(lines)
    new_lines = lines[:insert_at] + ["", fewshot_block.strip(), ""] + lines[insert_at:]
    return "\n".join(new_lines)
