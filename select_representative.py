"""
select_representative：
仅按 reward 选优 + 聚类保证多样性；输出动作为 [-1,1] 一位小数
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans


def obs_split(obs: List[float]) -> Tuple[int, List[float]]:
    """返回 (房间数 n, 温度向量 temps)"""
    n = max(1, (len(obs) - 2) // 3)
    return n, list(obs[:n])

def to_pm1_actions(action_f: List[float], n: int):

    if len(action_f) == 1 and n > 1:
        action_f = [action_f[0]] * n
    if len(action_f) != n:
        return None
    ax = np.asarray(action_f, dtype=float)
    ax = np.clip(ax, -1.0, 1.0)  # 保险夹紧
    return ax.tolist()

def temps_matrix_for_clustering(temps_list: List[List[float]]) -> np.ndarray:
    """
    聚类用特征矩阵（仅房间温度）。不同 n 时，统一截到最小 n；再 z-score。
    """
    n_min = min(len(t) for t in temps_list)
    T = np.stack([np.asarray(t[:n_min], dtype=np.float32) for t in temps_list], axis=0)
    mu = T.mean(axis=0, keepdims=True)
    sigma = T.std(axis=0, keepdims=True) + 1e-6
    return (T - mu) / sigma


def main(args):
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    if not traj:
        raise ValueError("Empty trajectory.")

    cand = []
    for st in traj:
        obs  = st.get("obs") or st.get("obs_before")
        actf = st.get("action")
        rew  = float(st.get("reward", 0.0))
        if obs is None or actf is None:
            continue
        n, temps = obs_split(obs)
        act_list = actf if isinstance(actf, list) else [actf]
        act_pm1 = to_pm1_actions(list(map(float, act_list)), n)
        if act_pm1 is None:
            continue
        cand.append((obs, temps, act_pm1, rew))

    if not cand:
        raise ValueError("No candidate samples.")

    # 先按 reward 预选 Top-K
    cand.sort(key=lambda x: x[3], reverse=True)
    pool = cand[: min(len(cand), args.preselect)]

    # 聚类（只看温度向量），保证多样性
    temps_list = [c[1] for c in pool]
    X = temps_matrix_for_clustering(temps_list)
    k = min(args.clusters, len(pool))
    if k <= 1:
        labels = np.zeros(len(pool), dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

    # 每个簇内按 reward 再取 Top-N
    chosen = []
    for cid in range(int(labels.max()) + 1):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        cluster_samples = [pool[i] for i in idxs]
        cluster_samples.sort(key=lambda x: x[3], reverse=True)
        chosen.extend(cluster_samples[: args.n_per_cluster])

    if not chosen:
        raise ValueError("No sample chosen after clustering.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    structured = [
        {
            "obs": obs,
            "actions": [float(f"{a:.1f}") for a in act_pm1],  # [-1,1] 一位小数
            "reward": float(rew),
            "building": args.building,
            "climate": args.climate,
            "location": args.location,
        }
        for (obs, _temps, act_pm1, rew) in chosen
    ]
    out_path = out / "few_shot_examples_structured.json"
    out_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote:", str(out_path))
    print(f"[stats] candidates={len(cand)}, pool={len(pool)}, clusters={k}, "
          f"n_per_cluster={args.n_per_cluster}, total_out={len(structured)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default="runs_officesmall_hotdry/ppo_trajectory.json")
    p.add_argument("--out_dir", default="fs_out")

    p.add_argument("--preselect", type=int, default=2000)
    p.add_argument("--clusters", type=int, default=12)
    p.add_argument("--n_per_cluster", type=int, default=20)
    p.add_argument("--building", default="OfficeSmall")
    p.add_argument("--climate",  default="Hot_Dry")
    p.add_argument("--location", default="Tucson")
    args = p.parse_args()
    main(args)
