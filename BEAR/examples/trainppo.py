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
# 'Tucson'：具体地理位置（美国图森市，典型干热气候）。
# BuildingEnvReal：BEAR项目中的真实建筑环境仿真器，基于输入参数生成可交互的环境（如温度动态、HVAC系统模型、能耗计算等）。
# make_vec_env：Stable Baselines3的工具函数，用于创建向量环境（支持多环境并行，此处n_envs=1为单环境）
# *
Parameter = ParameterGenerator('OfficeSmall', 'Hot_Dry', 'Tucson')
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
    model = PPO("MlpPolicy", None, verbose=1, learning_rate=3e-4, n_steps=2048, ent_coef=0.01, clip_range=0.2, n_epochs=10)
    model.set_env(vec_env)  # 显式绑定环境

# 添加调试输出
print(f"Model environment: {model.env}")
print(f"vec_env initialized: {vec_env}")
# 在 Step 1 后添加调试代码
print("vec_env 是否初始化成功:", vec_env is not None)  # 应输出 True
print("vec_env 类型:", type(vec_env))  # 应输出 <class 'stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv'>

# 在环境交互测试前添加
print("环境动作空间类型:", vec_env.action_space)  # 应输出 Box（连续）或 Discrete（离散）
print("环境动作空间形状:", vec_env.action_space.shape)  # 关键：输出如 (10,) 表示10维连续动作


# 在 Step 1 后添加测试代码
try:
    test_obs = vec_env.reset()  # 重置环境，获取初始状态
    print("初始状态 shape:", test_obs.shape)  # 应输出状态维度（如 (7,) 对应7个区域的温度）

    # 随机生成一个动作（假设动作空间是连续的）
    test_action = vec_env.action_space.sample()
    test_next_obs, test_reward, test_done, test_info = vec_env.step(test_action)
    print("动作执行成功，下一步状态 shape:", test_next_obs.shape)
except Exception as e:
    print("环境交互失败，错误信息:", e)
# 训练模型
model.learn(total_timesteps=20000)



# 保存模型
model.save("PPO_quick")
