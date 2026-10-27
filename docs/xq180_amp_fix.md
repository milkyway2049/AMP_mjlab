# XQ180 AMP 训练修复

2026-10-22，针对 [user_pd_v5 日志分析](xq180_audit/user_pd_v5_log_analysis/analysis.md)中确认的实现问题完成修复。PD 继续使用用户指定的 `user_pd_v5`，动作库仍为 17 段 `WalkandRun`，训练版本为 `ampfix_v6`。

## 修复内容

- AMP running mean/variance 使用保留的原始 policy/expert 特征更新；归一化后的张量只用于判别器训练。
- 梯度惩罚与判别损失使用同一套归一化 expert 输入。
- policy 与 discriminator 分别裁剪梯度，默认范数上限均为 1。分布式同步同时覆盖判别器参数和梯度。
- 对输入、完整 PPO+AMP loss、梯度范数检查有限性。无效批次不会执行 optimizer step 或更新统计；整轮无有效批次时立即报错停止。
- 初次 rollout 也按训练观测归一化器处理，支持加载已有策略后正确开始第一步训练。
- 新增日志：`Loss/kl`、`Loss/clip_fraction`、`Loss/action_saturation_fraction`、`Loss/policy_grad_norm`、`Loss/amp_grad_norm`、`Loss/effective_updates`、`Loss/policy_updates`、`Loss/policy_kl_stopped`。梯度范数为裁剪前数值。
- XQ180 使用分阶段学习率：0–5999 轮为 0.0003，6000–19999 轮为 0.0001，20000 轮后为 0.00005。阶段按绝对训练迭代计数执行，续训同样适用；修改基础 learning rate 时各阶段按 1、1/3、1/6 倍缩放。
- 当观测到 KL 超过 `4 × desired_kl = 0.04`，停止当前 rollout 剩余 minibatch 的 PPO 参数更新，但继续该轮全部 AMP expert minibatch，避免只训练动作列表前几段。该检查发生在下一次参数更新前，不会回滚此前已发生的更新，因此不是严格的 KL 硬约束。
- fixed schedule 续训后会将配置要求的当前阶段学习率重新写入 optimizer 参数组，避免旧 checkpoint 的学习率覆盖新配置。

角速度奖励 std 的调整属于后续行为调参，尚需固定转向命令评估；本次优先修复已确认的数据处理错误及训练更新保护。

## 从已检查的旧模型继续训练

旧 checkpoint 的 AMP 统计不符合修复后的原始特征定义，不能直接完整恢复训练。旧模型仍可按原参数回放；训练 warm start 需显式指定 `--agent.amp-reset-on-load True`，保留 actor、critic、policy std 与观测归一化权重，重置 discriminator、AMP normalizer、AMP replay 和 optimizer 状态。

在仓库根目录的 `mjlab` 环境执行：

```bash
python scripts/train.py XQ180-AMP-Flat \
  --env.scene.num-envs=4096 \
  --checkpoint-file logs/rsl_rl/xq180_amp_locomotion_no_recovery_userpd_v5/2026-10-22_00-08-45/model_30000.pt \
  --agent.amp-reset-on-load True \
  --agent.max-iterations=70000 \
  --enable-nan-guard True
```

该命令从迭代计数 30000 开始，再运行 70000 轮，到迭代计数 100000 的总目标；学习率从该阶段的 0.00005 开始。它属于保留策略的 warm start，不等同于完整恢复旧优化过程。`--checkpoint-file` 允许指定其他实验目录中的具体文件。

新日志保存在独立目录：

```text
logs/rsl_rl/xq180_amp_locomotion_no_recovery_userpd_v5_ampfix_v6/
```

原训练进程缓存了旧代码，需要停止旧进程后执行新命令。修复后的 checkpoint 带有 `amp_normalizer_version=2`；之后从新 checkpoint 正常续训时省略 `amp-reset-on-load`，即可恢复与新方法匹配的 AMP 和 optimizer 状态。

## 从头训练

```bash
python scripts/train.py XQ180-AMP-Flat \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=100000 \
  --enable-nan-guard True
```

默认不加载旧模型，不需要重置标志。PD、action scale、历史观测及部署接口与 `user_pd_v5` 一致，训练数据继续排除爬起动作。当前训练目录和版本见 [接口 JSON](xq180_policy_io.json)。

## 验证与限制

回归覆盖原始特征统计、梯度惩罚输入一致性、policy/discriminator 梯度裁剪、非有限 loss/gradient 阻止优化、单个异常 batch 不污染后续有效更新、KL 停止只冻结 PPO、学习率阶段与 optimizer 恢复、旧 checkpoint 显式重置和新版本 checkpoint 兼容。

CPU 实际环境确认 Flat/Rough 的初始化、重置、步进及立即跌倒终止正常。另加载真实旧 run 的 `model_30000.pt`，完成两轮 PPO/AMP 更新（每轮 20 个 AMP minibatch），验证参数和新统计有限、跳过无效批次为 0、阶段学习率为 0.00005，并完成真实 checkpoint 保存、ONNX 导出和新格式重新加载。

这些测试使用少量 CPU 环境验证代码与兼容性，不代表 4096 环境 GPU 长期收敛。分布式同步覆盖有隔离测试，尚未进行多 GPU 实训验证。
