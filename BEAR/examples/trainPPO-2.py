from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from BEAR.Utils.utils_building import ParameterGenerator
from BEAR.Env.env_building import BuildingEnvReal
import numpy as np

# -------------------------- Step 1: 环境初始化（关键修复） --------------------------
# 功能：创建建筑环境仿真对象，配置模拟参数
Parameter = ParameterGenerator('OfficeSmall', 'Hot_Dry', 'Tucson')  # 建筑类型/气候/位置参数
vec_env = make_vec_env(
    lambda: BuildingEnvReal(Parameter),  # 环境生成函数
    n_envs=1,  # 单环境（可扩展为多环境并行）
    vec_env_cls=None,  # 使用默认的DummyVecEnv
    vec_env_kwargs=None
)

# 打印环境初始化状态（关键调试）
print("="*50)
print("vec_env 初始化状态:", "成功" if vec_env is not None else "失败")
print("vec_env 类型:", type(vec_env))  # 应输出 DummyVecEnv
print("="*50 + "\n")

# -------------------------- Step 2: 模型加载/创建（关键修复） --------------------------
# 功能：加载预训练模型或初始化新的PPO智能体（确保环境绑定）
try:
    # 尝试加载预训练模型（需确保模型与环境兼容）
    model = PPO.load("PPO_quick", env=vec_env)  # 显式传递环境
    print("成功加载预训练模型！")
except FileNotFoundError:
    # 未找到预训练模型时，创建新模型并显式绑定环境
    model = PPO(
        policy="MlpPolicy",  # 多层感知机策略网络
        env=vec_env,         # 显式传递环境（核心修复）
        verbose=1,           # 输出训练日志
        learning_rate=3e-4,  # 学习率
        n_steps=2048,        # 每轮更新经验步数
        ent_coef=0.01,       # 熵系数（防止策略过拟合）
        clip_range=0.2,      # 策略更新裁剪范围（稳定训练）
        n_epochs=10          # 每轮更新迭代次数
    )
    print("未找到预训练模型，创建新模型并绑定环境。")

# 验证模型与环境绑定（关键调试）
print("\n" + "="*50)
print("模型绑定的环境:", model.env)  # 应输出 vec_env 对象
print("模型动作空间类型:", model.action_space)  # 应与 vec_env.action_space 一致
print("模型动作空间形状:", model.action_space.shape)  # 应输出 (6,)（匹配环境）
print("="*50 + "\n")

# -------------------------- Step 3: 环境交互测试（关键调试） --------------------------
# 功能：验证环境与模型能否正常交互（状态/动作维度匹配）
try:
    # 重置环境获取初始状态
    test_obs = vec_env.reset()
    print("初始状态 shape:", test_obs.shape)  # 应输出 (1, 20)（1环境实例，20维状态）

    # 生成与环境动作空间匹配的随机动作（6维）
    # 错误：test_action = vec_env.action_space.sample()
    test_action = np.expand_dims(vec_env.action_space.sample(), axis=0)

    print("随机动作 shape:", test_action.shape)  # 应输出 (6,)

    # 执行动作并获取环境反馈
    test_next_obs, test_reward, test_done, test_info = vec_env.step(test_action)
    print("下一步状态 shape:", test_next_obs.shape)  # 应输出 (1, 20)（与环境状态维度一致）
    print("即时奖励:", test_reward)  # 应输出标量值（如 -0.5 表示舒适度惩罚）
    print("是否终止:", test_done)    # 应输出 False（24小时模拟不终止）
except Exception as e:
    print("\n环境交互失败，错误信息:", e)
    raise  # 抛出异常终止程序（避免后续训练无效）

# -------------------------- Step 4: 模型训练（核心流程） --------------------------
print("\n" + "="*50)
print("开始模型训练...")
model.learn(
    total_timesteps=20000,  # 总训练步数（可根据需求调整）
    log_interval=100,       # 日志输出间隔（每100步打印一次）  # 训练过程中自动保存模型（可选）
)
print("训练完成！")
model.save("PPO_quick_final")

# -------------------------- Step 5: 模型保存与可视化 --------------------------
# 保存最终模型
model.save("PPO_quick_final")
print("\n模型已保存至: PPO_quick_final.zip")

# 可选：可视化训练过程（需安装 tensorboard）
# from stable_baselines3.common import results_plotter
# results_plotter.plot_results(["PPO_quick"], 20000, results_plotter.X_TIMESTEPS, "Training Reward")
# plt.show()