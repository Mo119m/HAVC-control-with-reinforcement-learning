# 把轨迹翻成renlei语言的描述，用来拼提示词/日志
import json
import argparse
from typing import List
import numpy as np

# 背景一句话
def meta_translator(building: str, location: str, weather: str):
    return (
        f"You are the HVAC administrator responsible for managing a building of type "
        f"{building} located in {location}, where the climate is {weather}."
    )

# 根据室外温度和目标温度，简单给个模式提示）
def instruction_translator(outside_temp: float, target_temp: float):
    mode = "heating" if outside_temp < target_temp else "cooling"
    direction = "below" if mode == "heating" else "above"
    return (
        f"Currently the outside temperature ({outside_temp:.1f}°C) is {direction} "
        f"the target ({target_temp:.1f}°C), so switch to {mode} mode.\n"
        "Guidelines:\n"
        "1. Actions are integers in [-100, 100], one per zone.\n"
        "2. If room temperature > target, larger difference → lower action.\n"
        "3. If room temperature < target, larger difference → higher action.\n"
        "4. Small deadband is OK (±0.2°C) to avoid oscillation."
    )

# 当前状态--自然语言
def state_translator(obs: List[float], target_temp: float):
    roomnum = (len(obs) - 2) // 3
    zones   = obs[0:roomnum]
    outdoor = obs[roomnum]
    ghi     = obs[roomnum + 1 : 2 * roomnum + 1]
    ground  = obs[2 * roomnum + 1]
    extra   = obs[2 * roomnum + 2 : 3 * roomnum + 2]  

    lines = [f"Target temperature is {target_temp:.1f}°C."]
    for i, t in enumerate(zones, 1):
        lines.append(f"Room {i}: {t:.1f}°C.")
    lines.append(f"Outside temperature: {outdoor:.1f}°C.")
    if len(ghi) == roomnum:
        lines.append(f"Global Horizontal Irradiance (avg): {float(np.mean(ghi)):.1f} W.")
    lines.append(f"Ground temperature: {ground:.1f}°C.")
    if len(extra) == roomnum:
        for i, v in enumerate(extra, 1):
            lines.append(f"Zone {i} feature: {v:.3f}.")
    return "\n".join(lines)

# 动作翻译：自动识别尺度：
def action_translator(action: List[float]):
    if not action:
        return "Actions set to: []."
    is_small_scale = all(abs(a) <= 1.2 for a in action)
    if is_small_scale:
        ints = [int(round(max(-1.0, min(1.0, a)) * 100)) for a in action]
    else:
        # 已经是大尺度：直接裁到 [-100,100]，再取整
        ints = [int(round(max(-100.0, min(100.0, a)))) for a in action]
    return "Actions set to: [" + ", ".join(map(str, ints)) + "]."

# 反馈：根据下一帧（或终止帧）和目标温度，给出“加/减/保持”的建议
def feedback_translator(action: List[float], reward: float, next_obs: List[float],
                        target_temp: float, terminal_obs: List[float] = None, tol: float = 0.2):
    comments = [f"Reward: {reward:.2f}."]
    # done=True 且有 terminal_obs 时优先用它
    obs_for_eval = terminal_obs if (terminal_obs is not None and len(terminal_obs) > 0) else next_obs
    if not obs_for_eval:
        comments.append("No next observation available for feedback.")
        return " ".join(comments)

    roomnum = (len(obs_for_eval) - 2) // 3
    next_zones = obs_for_eval[:roomnum]
    for i, t in enumerate(next_zones, 1):
        delta = t - target_temp
        if delta > tol:
            comments.append(f"Room {i} is {t:.1f}°C (> {target_temp:.1f}+{tol}), decrease action.")
        elif delta < -tol:
            comments.append(f"Room {i} is {t:.1f}°C (< {target_temp:.1f}-{tol}), increase action.")
        else:
            comments.append(f"Room {i} is {t:.1f}°C (~ target ±{tol}), keep or set small action.")
    return " ".join(comments)

# 主流程：把每一步翻成“场景说明+当前状态+动作+反馈”
def translate_trajectory_all(input_path: str, output_path: str,
                             building: str, location: str, weather: str,
                             target_temp: float):
    with open(input_path, "r", encoding="utf-8") as f:
        traj = json.load(f)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(meta_translator(building, location, weather) + "\n\n")

        for i, step in enumerate(traj):
            obs         = step.get("obs", [])
            act         = step.get("action", [])
            rew         = step.get("reward", 0.0)
            next_obs    = step.get("next_obs", [])
            done        = step.get("done", False)
            terminal    = step.get("terminal_obs", None)

            roomnum = (len(obs) - 2) // 3 if obs else 0
            outside = obs[roomnum] if obs and roomnum < len(obs) else float("nan")

            out.write(f"- Step {i} -\n")
            out.write(instruction_translator(outside, target_temp) + "\n\n")
            out.write(state_translator(obs, target_temp) + "\n\n")
            out.write(action_translator(act) + "\n\n")
            out.write(feedback_translator(act, rew, next_obs, target_temp, terminal_obs=terminal) + "\n\n")
            if done:
                out.write("Episode ends here.\n\n")

    print(f"natural language at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs_officesmall_hotdry/ppo_trajectory.json")
    parser.add_argument("--output", default="runs_officesmall_hotdry/translated_full_prompt.txt")
    parser.add_argument("--building", default="OfficeSmall")
    parser.add_argument("--location", default="Tucson")
    parser.add_argument("--weather", default="Hot_Dry")
    parser.add_argument("--target", type=float, default=22.0)
    args = parser.parse_args()

    translate_trajectory_all(
        args.input, args.output, args.building,
        args.location, args.weather,
        args.target
    )
    print("done")
