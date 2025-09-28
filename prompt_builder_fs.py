# 将 few_shot_examples_structured.json 转为训练脚本可直接使用的 rollout-style JSON


import json
from textwrap import dedent
from typing import List, Dict, Any, Optional
import argparse
import os

DEFAULT_SRC = "fs_out/few_shot_examples_structured.json"
DEFAULT_OUT = "mini_rollout_from_fs.json"

def zone_count_from_obs(obs: List[float]):
    """
    观测向量结构约定：3n+2 = temps(n) + outside(1) + ghi(n) + ground(1) + occ(n)
    """
    try:
        n = (len(obs) - 2) // 3
        return n if n > 0 else 1
    except Exception:
        return 1

def _f1(x):
    return f"{float(x):.1f}"

def _split_camel(s: str):
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s).strip()

def _pretty_building(s: str):
    s = s.replace("_", " ").replace("-", " ").strip()
    s = _split_camel(s)
    return " ".join(w.capitalize() for w in s.split())

def _pretty_climate(s: str):
    s = s.strip().replace("_", " ")
    s = s.replace("Hot Dry", "Hot and Dry").replace("Hot Humid", "Hot and Humid").replace("Warm Humid", "Warm and Humid")
    return " ".join(w.capitalize() for w in s.split())

def extract_env_terms(obs: List[float], n_override: Optional[int] = None):
    n = n_override if (n_override is not None) else zone_count_from_obs(obs)
    outside = float(obs[n]) if len(obs) > n else 0.0
    ghi_vals = [float(x) for x in obs[n+1:2*n+1]] if len(obs) >= 2*n+1 else []
    ghi_avg = (sum(ghi_vals) / len(ghi_vals)) if ghi_vals else 0.0
    ground_idx = 2*n + 1
    ground = float(obs[ground_idx]) if len(obs) > ground_idx else 0.0
    occ_vals = [float(x) for x in obs[2*n+2:3*n+2]] if len(obs) >= 3*n+2 else []
    occ_sum_kw = sum(occ_vals) if occ_vals else 0.0   # 保留原始符号；展示时用绝对值
    return {"outside": outside, "ghi_avg": ghi_avg, "ground": ground, "occ_sum_kw": occ_sum_kw}

def build_prompt_bracket_only(
    obs: List[float],
    building: str,
    location: str,
    climate: str,
    target: float = 22.0,
    n_rooms: Optional[int] = None
):
    """
    生成【不含历史】的自然语言任务描述，要求模型只输出一行 JSON 数组（与 SYSTEM_MSG 对齐）。
    """
    n = n_rooms if (n_rooms is not None) else zone_count_from_obs(obs)
    temps = obs[:n]
    env = extract_env_terms(obs, n_override=n)

    building_s = _pretty_building(building)
    climate_s  = _pretty_climate(climate)
    temp_lines = "\n".join([f"   Room {i+1}: {_f1(temps[i])} degrees Celsius" for i in range(n)])
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
    1. Output one list of length {n} with each value in [-1, 1]. Positive = heating (raise temperature), negative = cooling (lower temperature).
    2. The order must match the room order above.
    3. Match the sign to (Target − Room). Avoid identical actions for all rooms unless all room temperatures are identical.
    IMPORTANT: Respond with exactly one single JSON array of floats:
    [x1, x2, ..., x{n}]
    Do not include any other text.
    """).strip()
    return prompt

# 主转换逻辑
def convert(src_path: str, out_path: str):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Input file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    entries = []
    for rec in data:
        obs = rec["obs"]
        actions = rec["actions"]

        building = rec.get("building", "OfficeSmall")
        climate  = rec.get("climate",  "Hot_Dry")
        location = rec.get("location", "Tucson")
        target   = float(rec.get("target", 22.0))
        reward   = float(rec.get("reward", 0.0))

        prompt = build_prompt_bracket_only(obs, building, location, climate, target=target)

        entries.append({
            "prompt": prompt,
            "used_fallback": False,
            "parsed_from": "actions_line",            
            "action_unit": [float(a) for a in actions],
            "reward": reward,
            "done": False,
        })

    if entries:
        entries[-1]["done"] = True

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return len(entries)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="输入结构化数据（list of records）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 rollout-style JSON（供训练脚本读取）")
    args = ap.parse_args()

    n = convert(args.src, args.out)
    print(f"Wrote {n} samples {args.out}")
    print("在训练脚本中设置 ROLLOUT_GLOBS 指向该文件，例如：")
    print(f'ROLLOUT_GLOBS="{args.out}"')

if __name__ == "__main__":
    main()
