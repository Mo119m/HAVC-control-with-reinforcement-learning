print("==== THIS IS THE CORRECT VERSION ====")
import os
print("=== Running testppo from:", os.path.abspath(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from BEAR.Utils.utils_building import ParameterGenerator
from BEAR.Env.env_building import BuildingEnvReal

# Step 1: Create environment

# 功能：创建建筑环境仿真对象，配置模拟参数。
# 关键细节：
# ParameterGenerator：来自BEAR工具包的参数生成器，用于定义建筑类型、气候类型和地理位置。
# 'OfficeSmall'：模拟建筑类型（小型办公楼）。
# 'Hot_Dry'：气候类型（炎热干燥）。
# 'Tucson'：具体地理位置ls（美国图森市，典型干热气候）。
# BuildingEnvReal：BEAR项目中的真实建筑环境仿真器，基于输入参数生成可交互的环境（如温度动态、HVAC系统模型、能耗计算等）。
# make_vec_env：Stable Baselines3的工具函数，用于创建向量环境（支持多环境并行，此处n_envs=1为单环境）
# *
Parameter = ParameterGenerator('OfficeSmall', 'Hot_Dry', 'Tucson', root='/Users/Mo/Desktop/BEAR/BEAR/Data/')
vec_env = make_vec_env(lambda: BuildingEnvReal(Parameter), n_envs=1)

# Step 2: Load or create model
# 功能：加载预训练模型或初始化新的PPO智能体。
# 关键细节：
# PPO算法：Stable Baselines3实现的近端策略优化算法（Proximal Policy Optimization），适合连续/离散动作空间的强化学习任务（此处建筑调控可能为连续动作，如设定温度值）。
# 策略网络："MlpPolicy"表示使用多层感知机（MLP）作为策略网络，输入为环境状态（如各区域温度、湿度），输出为动作（如HVAC系统的设定参数）。
# 超参数：
# learning_rate=3e-4：学习率（平衡训练稳定性和收敛速度）。
# n_steps=2048：每轮更新使用的经验步数（PPO的“批量大小”）。
# clip_range=0.2：策略更新的裁剪范围（防止策略剧烈变化，保证训练稳定性）。
try:
    model = PPO.load("PPO_quick")
except FileNotFoundError:
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=2048, ent_coef=0.01, clip_range=0.2, n_epochs=10)



# Step 3: Run simulation
# 功能：运行24小时的仿真（模拟一天内的温度调控过程），记录状态、动作和奖励数据。
# 关键细节：
# vec_env.reset()：重置环境到初始状态（如初始温度分布、HVAC系统初始设定）。
# model.predict(obs)：智能体根据当前环境状态obs预测最优动作（如各区域的温度设定值）。
# vec_env.step(action)：执行动作，环境返回新的状态obs、即时奖励rewards、是否终止dones（此处dones=False，因模拟24小时不终止）和额外信息info（如能耗、舒适度指标）。
# 数据记录：statelist存储每小时的环境状态（推测为各区域温度），actionlist存储智能体输出的动作（如HVAC设定值），rewardlist存储即时奖励（可能基于温度舒适度、能耗等设计）。
obs = vec_env.reset()
rewardlist = []
statelist = []
actionlist = []
# [加载预训练模型] → [初始化环境] → [循环预测动作] → [执行动作并收集反馈]
#        ↓                                 ↓
#   model.predict()                    vec_env.step()
#        ↓                                 ↓
#      [记录 statelist, actionlist, rewardlist]

for i in range(24):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    statelist.append(obs.copy())
    actionlist.append(action.copy())
    rewardlist.append(rewards)


# Step 4: Visualization
import numpy as np
import matplotlib.pyplot as plt
#  1. 评估模型性能

# 观测数据 (statelist)：
    # 可用于分析建筑各区域温度随时间的变化趋势。
    # 判断是否满足设定的目标温度范围（如舒适度要求）。
    # 示例：判断模型是否成功将室内温度维持在 [18°C, 22°C] 舒适区间。
# 动作数据 (actionlist)：
    # 显示 HVAC 系统的控制策略（如制冷、制热强度）。
    # 分析能耗情况，判断是否存在频繁切换或过度控制。
    # 示例：查看是否有不必要的高功率运行时段。
# 奖励数据 (rewardlist)：
    # 反映每一步控制效果的好坏。
    # 奖励值越高表示控制越优（节能+舒适兼顾）。
    # 示例：奖励曲线下降可能意味着控制策略失效或环境变化剧烈。
    # 转换 statelist 为二维数组 (24, 7)
temperature_data = np.array(statelist).squeeze()
action_data = np.array(actionlist).squeeze() if len(actionlist) > 0 else None
reward_data = np.array(rewardlist).squeeze() if len(rewardlist) > 0 else None

# 创建 2x2 子图
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(' RL control modal based on Bear', fontsize=16)

# 图表 1: 温度变化曲线
axs[0, 0].plot(temperature_data[:, :7])
axs[0, 0].set_title('state：Temperature Change Over Time (Hours) ')
axs[0, 0].set_xlabel('Hours')
axs[0, 0].set_ylabel('Temperature (°C)')
axs[0, 0].legend(['South', 'East', 'North', 'West', 'Core', 'Plenum', 'Outside'], loc='lower right')
axs[0, 0].grid(True)

# 图表 2: HVAC 功率消耗
if action_data is not None:
    if len(action_data.shape) == 1:
        axs[0, 1].plot(action_data, label='HVAC Action')
    else:
        for i in range(action_data.shape[1]):
            axs[0, 1].plot(action_data[:, i], label=f'HVAC Action {i+1}')
    axs[0, 1].set_title('Action：HVAC Power Consumption Over Time ')
    axs[0, 1].set_xlabel('Hours')
    axs[0, 1].set_ylabel('Power (Watts)')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True)
else:
    fig.delaxes(axs[0, 1])  # 如果没有数据则删除该子图
    print("actionlist 为空，跳过绘制动作图")

# 图表 3: 奖励变化曲线
if reward_data is not None:
    axs[1, 0].plot(reward_data, color='green', label='Episode Reward')
    axs[1, 0].set_title('Training Rewards Over Episodes')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Reward')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(True)
else:
    fig.delaxes(axs[1, 1])  # 如果没有数据则删除该子图
    print("rewardlist 为空，跳过绘制奖励图")

# 图表 4: 初始状态可视化（可选）
try:
    initial_state = vec_env.reset()
    if initial_state is not None:
        axs[1, 1].bar(range(len(initial_state[0])), initial_state[0])
        axs[1, 1].set_title('Initial State Values')
        axs[1, 1].set_xlabel('State Index')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].grid(True)
    else:
        fig.delaxes(axs[1, 1])
        print("初始状态数据不可用，跳过绘制初始状态图")
except Exception as e:
    fig.delaxes(axs[1, 1])
    print(f"无法获取初始状态数据：{e}")

# 自动调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

