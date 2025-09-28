from textwrap import dedent
from typing import List, Dict, Any, Optional
import re

__all__ = ["zone_count_from_obs", "extract_env_terms", "build_prompt"]

DEBUG = True
def dprint(*a, **k):
    if DEBUG: print(*a, **k)

def zone_count_from_obs(obs: List[float]) -> int:
    # 环境是 3n+2: temps(n), outside(1), ghi_per_zone(n), ground(1), occ_per_zone(n)
    try:
        n = (len(obs) - 2) // 3
        return n if n > 0 else 1
    except Exception:
        return 1

def _f1(x) -> str:
    return f"{float(x):.1f}"

def _fmt_list(xs: List[float]):
    return "[" + ", ".join(_f1(v) for v in xs) + "]"

def _split_camel(s: str):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s).strip()

def _pretty_building(s: str):
    s = s.replace("_", " ").replace("-", " ").strip()
    s = _split_camel(s)
    return " ".join(w.capitalize() for w in s.split())

def _pretty_climate(s: str):
    s = s.strip().replace("_", " ")
    s = s.replace("Hot Dry", "Hot and Dry").replace("Hot Humid", "Hot and Humid").replace("Warm Humid", "Warm and Humid")
    return " ".join(w.capitalize() for w in s.split())

def _history_lines(history: List[Dict[str, Any]], n: int) -> str:
    if not history:
        return "None"
    rows = []
    for h in history:
        step   = int(h.get("step", 0))
        a      = h.get("action", [])
        r      = float(h.get("reward", 0.0))
        envt   = h.get("env_temp", None)
        before = (h.get("obs_before", []) or [])[:n]
        after  = (h.get("obs_after",  []) or [])[:n]
        power  = h.get("power", 0)
        a_str  = "[" + ", ".join(f"{float(x):.2f}".rstrip("0").rstrip(".") for x in a) + "]"
        rows.append(
            f"Step {step}, Action: {a_str}, Reward: {r:.6g}, "
            f"Env Temp: {(_f1(envt) if envt is not None else 'N/A')}, "
            f"Room Temp Before: {_fmt_list(before)}, "
            f"Room Temp After:{_fmt_list(after)}, Power: {power}"
        )
    return "\n".join(rows)

def extract_env_terms(obs: List[float], n_override: Optional[int] = None) -> Dict[str, float]:
    # 3n+2 切片；允许外部传入 n 强制对齐
    n = n_override if (n_override is not None) else zone_count_from_obs(obs)
    outside = float(obs[n]) if len(obs) > n else 0.0
    ghi_vals = [float(x) for x in obs[n+1:2*n+1]] if len(obs) >= 2*n+1 else []
    ghi_avg = (sum(ghi_vals) / len(ghi_vals)) if ghi_vals else 0.0
    ground_idx = 2*n + 1
    ground = float(obs[ground_idx]) if len(obs) > ground_idx else 0.0
    occ_vals = [float(x) for x in obs[2*n+2:3*n+2]] if len(obs) >= 3*n+2 else []
    occ_sum_kw = sum(occ_vals) if occ_vals else 0.0  
    return {"outside": outside, "ghi_avg": ghi_avg, "ground": ground, "occ_sum_kw": occ_sum_kw}

def build_prompt(
    obs: List[float],
    building: str,
    location: str,
    climate: str,
    target: float,
    round_idx: int,
    history: List[Dict[str, Any]],
    history_lines: int = 1,
    n_rooms: Optional[int] = None,  
):
    n = n_rooms if (n_rooms is not None) else zone_count_from_obs(obs)
    temps = obs[:n]
    env = extract_env_terms(obs, n_override=n)

    building_s = _pretty_building(building)
    climate_s  = _pretty_climate(climate)

    temp_lines = "\n".join([f"   Room {i+1}: {_f1(temps[i])} degrees Celsius" for i in range(n)])
    hist_text  = _history_lines(history[-history_lines:] if history_lines > 0 else [], n)

    occ_kw_text = _f1(abs(env['occ_sum_kw'])) + " KW  (internal heat gain)"

    prompt = dedent(f"""\
    You are the HVAC administrator responsible for managing a building of type {building_s} located in {location}, where the climate is {climate_s}.
    The building has {n} rooms in total.
    Currently, temperature in each room is as follows:
{temp_lines}
    The external climate conditions are as follows:
       Outside Temperature: {_f1(env['outside'])} degrees Celsius.
       Global Horizontal Irradiance: {_f1(env['ghi_avg'])}
       Ground Temperature: {_f1(env['ground'])} degrees Celsius
       Occupant Power: {occ_kw_text}
       Target Temperature: {_f1(target)} degrees Celsius
    To optimize HVAC control, follow these:
    1. Output one list of length {n} with each number value ranging from -1 to 1. Absolute value means HVAC Power, positive number means heating(raise temperature) and negative numbers means cooling(lower temperature)
    2. The order must match the room order above.
    3. Match the sign to (Target − Room). Avoid identical actions for all rooms unless all room temperatures are identical.
    4. Since all actions are within the range [-1, 1], avoid making large changes compared to the most recent action history on this scale (e.g., a change of 0.5 is already significant), unless there is a notable change in room temperatures.

    History Action And Feedback Reference:
    {hist_text}

    IMPORTANT: Give 1–2 sentences of reasoning (no '[' or ']'). Then END with exactly one final line:
    Actions: [x1, x2, ..., x{n}]
    This must be the last line; 'Actions:' appears only once; after the final ']' output nothing else.


    """).strip()

    if DEBUG:
        dprint(f"[builder_control] n={n}, prompt_bytes={len(prompt.encode('utf-8'))}")
    return prompt
