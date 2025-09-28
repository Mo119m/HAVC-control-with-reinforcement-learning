### 基于 *PRE-TRAINED LARGE LANGUAGE MODELS FOR INDUSTRIAL CONTROL* 和 *BEAR* 仿真器思路的融合与添加进行对QWEN2.5-1.5B的微调。
目前正在修改finetune.py文件，研究如何融合 *Proximal Policy Optimization Algorithms* 论文的思路和公式在这一步。




这是一个完整的基于强化学习和大语言模型（LLM）的建筑能源管理系统训练和评估流程， 实现了从数据收集、样本选择、模型微调到性能评估的完整闭环， 可以系统地比较不同策略在建筑能源管理任务上的表现。
主要包括以下几个阶段：

1. 数据收集阶段
[ppo_collect.py] → ppo_trajectory.json
    使用PPO算法训练智能体，并收集训练过程中的轨迹数据
    轨迹数据保存在ppo_trajectory.json文件中

2. 样本选择阶段
[select_representative.py] → few_shot_examples_structured.json

    从轨迹数据中选择具有代表性的样本作为few-shot示例
    使用奖励值和聚类方法确保样本的质量和多样性
    输出结构化的few-shot示例文件

3. 微调数据生成阶段

[rollout_fewshot_version.py] → mini_rollout_fewshot.json
使用原始LLM模型结合few-shot示例生成控制动作
收集这些交互数据用于后续模型微调

4. 模型微调阶段
[7b_finetune.py] → 微调后的模型
使用生成的交互数据对LLM模型进行LoRA微调
微调过程中使用PPO算法优化模型策略
5. 模型评估阶段
5.1 生成评估数据
[7Blora_rollout.py] → mini_rollout_fewshot_7B_finetuned.json  (微调模型)
[only_history_rollout.py] → mini_rollout_llm_good.json  (仅历史信息)
5.2 可视化性能对比

[draw_reward.py] → 可视化三种策略的性能对比

三种控制策略对比
   （1）Few-shot + 微调模型: mini_rollout_fewshot_7B_finetuned.json
        使用经过微调的LLM模型
        结合few-shot示例进行决策
    （2）Few-shot + 原始模型: mini_rollout_fewshot.json
        使用原始的LLM模型（未微调）
        结合few-shot示例进行决策
    （3）仅历史信息: mini_rollout_llm_good.json
        使用原始LLM模型
        仅依赖历史交互信息进行决策





