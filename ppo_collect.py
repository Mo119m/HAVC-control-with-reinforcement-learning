# 跑一个 PPO 来“边训练边采样”，把轨迹用回调写到 JSON
import os
import json
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator
from recorder_v2 import TrajectoryRecorder  

building, weather, location = "OfficeSmall", "Hot_Dry", "Tucson"
data_root = "/Users/Mo/Desktop/BEAR/BEAR/Data/"

save_dir = "./runs_officesmall_hotdry"
os.makedirs(save_dir, exist_ok=True)

traj_path = os.path.join(save_dir, "ppo_trajectory.json")   # 轨迹文件
model_path = os.path.join(save_dir, "ppo_officesmall_hotdry.zip")

total_steps = 500000   
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# 环境这块：就是单环境（n_envs=1）  多环境的话需要再研究
param = ParameterGenerator(building, weather, location, root=data_root)
env = make_vec_env(lambda: BuildingEnvReal(param), n_envs=1)

print("vec action_space:", env.action_space)
print("low:",  env.envs[0].action_space.low)
print("high:", env.envs[0].action_space.high)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    seed=seed,
    device=device,
)
callback = TrajectoryRecorder(save_path=traj_path, verbose=1)

try:
    model.learn(total_timesteps=total_steps, callback=callback)
except KeyboardInterrupt:
    print("[ppo_collect]")

model.save(model_path)
print(f"[ppo_collect] done. model  {model_path}\ntrajectory -> {traj_path}")
