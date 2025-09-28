# 在不修改 build_prompt 的前提下，自动注入“相似+高reward”的 few-shot
import os, json
import numpy as np
from collections import deque
from stable_baselines3.common.env_util import make_vec_env

from BEAR.Utils.utils_building import ParameterGenerator
from BEAR.Env.env_building import BuildingEnvReal
from prompt_builder_control import build_prompt, extract_env_terms, zone_count_from_obs
from llm_agent_colab import call_llm, parse_actions

#  新增：few-shot 自动选择 & 注入 
from few_shot_auto import load_examples, select_examples, format_few_shot_block, inject_few_shot

BUILDING  = "OfficeSmall"
CLIMATE   = "Hot_Dry"
LOCATION  = "Tucson"
TARGET    = 22.0

MAX_STEPS = 20
HIST_KEEP = 6
HIST_LINES_IN_PROMPT = 3
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

FEWSHOT_JSON = os.getenv("FEWSHOT_JSON", "/content/drive/MyDrive/BEAR_LLM/fs_out/few_shot_examples_structured.json")
K_FEWSHOT = int(os.getenv("K_FEWSHOT", "3"))         # 每步插入的示例条数
FEWSHOT_ALPHA = float(os.getenv("FEWSHOT_ALPHA", "0.6"))  # 0=只看reward，1=只看相似度

DEBUG = True
def dprint(*a, **k):
    if DEBUG: print(*a, **k)

def _outside_from_obs(obs):
    return extract_env_terms(obs)["outside"]

def main():
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    # 预加载 few-shot 数据集（不用则跳过注入）
    try:
        EX_DATASET = load_examples(FEWSHOT_JSON)
        dprint(f"[fewshot] loaded {len(EX_DATASET)} examples")
    except Exception as e:
        EX_DATASET = None
        dprint(f"[fewshot] skip (reason: {e})")

    param = ParameterGenerator(BUILDING, CLIMATE, LOCATION, root=DATA_ROOT, target=TARGET)
    vec_env = make_vec_env(lambda: BuildingEnvReal(param), n_envs=1)
    env = vec_env.envs[0]

    dprint("[env] action_space.low/high =", env.action_space.low, env.action_space.high)

    reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
    obs = np.array(obs).tolist()

    history = deque(maxlen=HIST_KEEP)
    logs = []

    for step in range(MAX_STEPS):
        n = zone_count_from_obs(obs)
        prompt = build_prompt(
            obs=obs, building=BUILDING, location=LOCATION, climate=CLIMATE,
            target=TARGET, round_idx=step+1,
            history=list(history), history_lines=HIST_LINES_IN_PROMPT
        )

        if EX_DATASET:
            exs = select_examples(
                EX_DATASET, current_obs=obs, k=K_FEWSHOT, alpha=FEWSHOT_ALPHA,
                building=BUILDING, climate=CLIMATE, location=LOCATION
            )
            few_block = format_few_shot_block(exs, target=TARGET, n=n)
            prompt = inject_few_shot(prompt, few_block)
        else:
            few_block = None

        raw_text = call_llm(
            prompt, n_actions=n, model_name=MODEL_NAME,
            max_new_tokens=256, temperature=0.3
        )

        action_unit, meta = parse_actions(raw_text, n)
        if action_unit is None:
            action_unit = [0.0]*n

        action_env = action_unit
        dprint(f"[step {step}] act_unit={np.round(action_unit,3).tolist()}")

        step_ret = env.step(action_env)
        if len(step_ret)==5:
            obs_next, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs_next, reward, done, info = step_ret
        obs_next = np.array(obs_next).tolist()

        history.append({
            "step": step+1,
            "action": [float(x) for x in action_unit],
            "reward": float(reward),
            "env_temp": _outside_from_obs(obs),
            "obs_before": obs,
            "obs_after":  obs_next,
            "power": (info or {}).get("power", None)
        })

        logs.append({
            "step": step,
            "prompt": prompt,
            "few_shot": few_block or "",
            "llm_raw": raw_text,
            "parsed_from": (meta or {}).get("parsed_from","failed"),
            "action_unit": [float(x) for x in action_unit],
            "action_env": action_env,
            "reward": float(reward),
            "done": bool(done),
            "obs": obs,
            "next_obs": obs_next,
            "env_temp": _outside_from_obs(obs),
        })

        obs = obs_next
        if done: break

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Saved: {SAVE_PATH}")

if __name__ == "__main__":
    main()
