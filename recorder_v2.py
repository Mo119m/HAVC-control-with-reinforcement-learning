# recorder_v2.py — 记录轨迹（动作裁剪后再存，同时保留 raw；修复 terminal_obs 的 or 报错）
import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrajectoryRecorder(BaseCallback):
    def __init__(self, save_path="ppo_trajectory.json", verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.buf = []
        self.prev_obs = None

    @staticmethod
    def _to_list(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if hasattr(x, "tolist"):
            v = x.tolist()
            return v if isinstance(v, list) else [float(v)]
        try:
            return [float(x)]
        except Exception:
            return list(x)

    def _grab_curr_obs(self):
        try:
            state = self.training_env.get_attr("state")[0]
            return state.tolist() if hasattr(state, "tolist") else list(state)
        except Exception:
            pass
        cand = self.locals.get("new_obs", None)
        if cand is None:
            cand = self.locals.get("observations", None)
        if cand is None:
            raise RuntimeError("拿不到当前观测（env.state/new_obs/observations 都为空）")
        v = cand[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def _on_rollout_start(self):
        self.prev_obs = self._grab_curr_obs()
        return True

    def _on_step(self):
        # 1) 原始动作（策略输出，可能越界）
        raw = self.locals["actions"][0]
        raw_arr = np.asarray(raw, dtype=float)
        action_raw = self._to_list(raw)

        # 2) 裁剪到环境动作空间（真实执行）
        try:
            low  = np.asarray(self.training_env.action_space.low, dtype=float)
            high = np.asarray(self.training_env.action_space.high, dtype=float)
        except Exception:
            low  = -np.ones_like(raw_arr, dtype=float)
            high =  np.ones_like(raw_arr, dtype=float)
        clipped = np.clip(raw_arr, low, high)
        action = clipped.tolist()

        # 3) 奖励/结束标志/info
        reward = float(self.locals["rewards"][0])
        done   = bool(self.locals["dones"][0])
        infos  = self.locals.get("infos", [{}])

        # 4) 下一观测（done=True 时通常是 reset 后的第一帧）
        next_obs = self._grab_curr_obs()

        # 5) 终止前观测（⚠️ 显式判断 None，不能用“or”）
        terminal_obs = None
        if done:
            # infos 可能是 list[dict] 或 dict
            info0 = None
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                info0 = infos[0]
            elif isinstance(infos, dict):
                info0 = infos
            if isinstance(info0, dict):
                term = info0.get("terminal_observation", None)
                if term is None:
                    term = info0.get("final_observation", None)
                if term is not None:
                    terminal_obs = term.tolist() if hasattr(term, "tolist") else list(term)

        # 6) 入缓存
        self.buf.append({
            "obs": self.prev_obs,
            "action_raw": action_raw,   # 调试用，可能越界
            "action": action,           # 已裁剪，真实执行
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "terminal_obs": terminal_obs
        })

        # 7) 推进
        self.prev_obs = next_obs
        return True

    def _on_training_end(self):
        d = os.path.dirname(self.save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.buf, f, indent=2)
        if self.verbose:
            print(f"[TrajectoryRecorder] wrote {len(self.buf)} steps to {self.save_path}")
