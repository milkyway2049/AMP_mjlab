# XQ180：仅行走训练与当前 PD

当前 PD：`user_pd_v5`，训练修复版本：`ampfix_v6`，更新日期 2026-10-22。AMP 训练只使用 `amp_ik/WalkandRun/` 中 17 段走跑、转向动作，共 7417 帧、148.34 秒。`Recovery/` 数据保留供历史对比，但不会进入判别器或环境重置。

## 当前用户 PD

采用用户指定的整数表，左右相同；shoulder pitch 一行按 Kp=50、Kd=2 处理。按关节名称绑定执行器，并根据新 Kp 重算 action scale。

| 关节组 | Kp | Kd |
|---|---:|---:|
| hip yaw | 200 | 10 |
| hip roll | 400 | 20 |
| hip pitch | 300 | 15 |
| knee | 300 | 15 |
| ankle pitch | 100 | 5 |
| waist yaw | 100 | 5 |
| shoulder pitch | 50 | 2 |
| shoulder roll | 50 | 2 |
| shoulder yaw | 25 | 1 |
| elbow | 25 | 1 |

当前完整参数见 [接口 JSON](xq180_policy_io.json)。此前 G1 构型推导方案保存在 [v4 参数快照](xq180_policy_io_g1_chain_v4.json)。以下推导及误差数据属于 v4 历史对照，`derive_xq180_pd.py` 只生成参考报告，不会切换当前用户 PD。

## 关节顺序与历史 v4 PD 推导

两台机器人的髋关节串联顺序不同：

```text
G1:    root → hip pitch → hip roll → hip yaw → knee → ankle pitch → ankle roll
XQ180: root → hip yaw   → hip roll → hip pitch → knee → ankle pitch
```

因此不能按照 G1 的数组下标给 XQ180 分配 PD，也不能只按整机质量或单个电机质量缩放。以左侧为例，XQ180 hip yaw 下游质量约 19.93 kg，G1 hip yaw 下游质量约 4.32 kg；hip pitch 下游分别约 12.51 kg 和 7.19 kg。惯量还取决于旋转轴、质心距离、当前姿态，不能直接把这些质量比当作最终增益倍数。

计算器分别加载两台机器人真实的父子连杆链、关节坐标、质量、质心、惯量和 armature。从每段当前 IK 走跑参考及对应 G1 参考中均匀抽取 32 帧，共 544 对姿态：

1. 摆动腿、腰和手臂：使用浮动根固定时完整质量矩阵 `M(q)` 的关节对角项。对角项对应其余关节速度为零的局部惯量。
2. 支撑腿：按参考库的左右 `foot_contact` 标记，将本侧足底六维速度固定，用 `J_base * v_base + J_joint * dq = 0` 消去浮动基座速度，计算 `M_support = T.T @ M @ T`，取对应关节对角项。
3. 在每个姿态对同组左右关节的 `I_XQ180 / I_G1` 取平均，再在全部姿态中取 P90 作为该组的参考倍数。P90 是覆盖较高负载姿态的工程选择，并非求解出的最优分位点。
4. 同时缩放 G1 的 Kp、Kd，再四舍五入取整数，最小值为 1：

```text
r = P90(mean_left_right(I_XQ180(q, mode) / I_G1(q, mode)))
Kp_XQ180 = max(1, round(r * Kp_G1))
Kd_XQ180 = max(1, round(r * Kd_G1))
```

同时按惯量比例缩放两项，在单关节二阶近似下保持 `sqrt(Kp/I)` 和 `Kd/(2*sqrt(Kp*I))` 接近 G1。实际动力学存在耦合、接触、重力和限幅，整数化也会改变这个近似，尤其是手臂小阻尼值。

| 关节组 | 所用惯量比 P90 | Kp | Kd |
|---|---:|---:|---:|
| hip yaw | 4.046 | 163 | 10 |
| hip roll | 3.903 | 387 | 25 |
| hip pitch | 3.120 | 309 | 20 |
| knee | 3.120 | 309 | 20 |
| ankle pitch | 3.523 | 100 | 6 |
| waist yaw | 1.731 | 70 | 4 |
| shoulder pitch | 1.021 | 15 | 1 |
| shoulder roll | 1.460 | 21 | 1 |
| shoulder yaw | 1.304 | 19 | 1 |
| elbow | 0.736 | 10 | 1 |

左右对称，单位分别为 Nm/rad、Nm·s/rad。代码按 XQ180 的**关节名称**设置执行器增益；关节数组仍采用 XQ180 实际顺序。动作 scale 根据新 Kp 重算，并以 G1 对应轴的角度幅度为上限。[v4 接口快照](xq180_policy_io_g1_chain_v4.json) 包含该阶段 19 关节的完整顺序、PD、scale；[推导 JSON](xq180_audit/g1_chain_integer_pd.json) 包含每个关节的父链、局部轴、名义世界轴、下游质量及分位数，便于复核。

```bash
python scripts/derive_xq180_pd.py
```

这组参数替代之前仅按名义站姿计算的 `g1_inertia_integer_v3`，并未承诺比原用户表具有更小的所有轨迹误差。当前 XQ180 armature=0.01 仍是模型假设；没有真实驱动器辨识、双支撑接触力或摩擦锥求解，需进一步通过闭环训练评估。

固定根、关闭重力和接触的整库位置跟踪检查中，v4 方案的走跑 RMS 误差为 0.14250 rad，原用户表为 0.12311 rad。该对照说明按模型动力学相似性计算的增益不等于轨迹误差最优增益，不能宣称本次 PD 已提升收敛或平衡。该阶段逐段数据见 [locomotion_servo_comparison.json](xq180_retarget/locomotion_servo_comparison.json)。

## 只训练站立、行走和转向

同时关闭了恢复动作的三个入口：AMP 专家根目录指向 `WalkandRun/`；reset 只从该目录抽样；`recovery_dir=None`、延迟环境比例为 0、最大延迟步数为 0。训练和回放配置一致。身体过低或倾倒时立即终止，不给躺地状态额外 5 秒的爬起时间。

| 参数 | 当前设置 | 目的 |
|---|---|---|
| 跌倒条件 | 根部相对地形 tile 原点高度 <0.75 m，或倾斜 >50° | 尽早结束跌倒 episode |
| 正常高度奖励 | 目标 1.05 m，std 0.12，权重 0.5 | 替代原本仅对恢复环境生效的高度项 |
| 初期命令 vx/vy/wz | `[-0.3,0.8] / ±0.2 / ±0.5` | 先学习较慢行走 |
| 约 2000 轮后 | `[-0.5,1.2] / ±0.3 / ±0.75` | 逐步扩大范围 |
| 约 6000 轮后 | `[-0.8,1.8] / ±0.5 / ±1.0` | 后期覆盖较快行走 |
| 命令重采样、站立比例 | 4–8 秒、10% | 留出平衡和站立练习 |
| 推扰 | 每 6–10 秒，vx ±0.3、vy ±0.2，无垂向和角速度推扰 | 避免频繁击倒 |
| 随机摩擦、腰部质心偏置 | 0.5–1.0、各轴 ±1 cm | 初期保留适度随机化 |
| Rough 初始地形等级上限 | 1 | 从较低难度开始 |
| PPO 学习率 | 0–5999 轮 0.0003；6000–19999 轮 0.0001；之后 0.00005 | 后期减小更新幅度 |
| PPO KL 停止阈值 | 0.04 | 超限后停止本轮后续 PPO 更新，继续 AMP 更新 |
| 初始动作噪声 std | 0.5 | 减少初期动作幅度 |
| 执行动作裁剪 | `[-3,3]`，裁剪后动作写入历史观测 | 限制动作反馈幅度 |
| MuJoCo solver iterations | 20，dt 5 ms，控制 50 Hz | 增加接触求解迭代 |

当前参考帧根高最低约 0.927 m、最大倾斜约 27.43°，均留有终止阈值余量。Rough 的高度以 tile 原点为参考，并非实时局部足底地面高度。速度课程按全局控制步计数，不是按奖励达标触发；换 rollout 长度时应同步调整轮数换算。前期命令较慢，但专家库仍包含全部走跑参考，并非只保留慢走。

## AMP 训练修复

已修复原始特征统计、梯度惩罚尺度、判别器梯度裁剪，并增加 KL 和梯度诊断。旧 checkpoint 训练需显式重置 AMP 状态；从 `model_30000.pt` 跑剩余 70000 轮的命令见 [修复及续训说明](xq180_amp_fix.md)。

## 启动与检查

在仓库根目录、`mjlab` 环境中执行：

```bash
# 使用新实验目录从头训练；不恢复之前包含爬起或不同 PD 的 checkpoint。
python scripts/train.py XQ180-AMP-Flat --env.scene.num-envs=4096 --agent.max-iterations=100000 --enable-nan-guard True

# 连续慢速播放当前训练使用的 17 段动作。
python scripts/play_motions.py --speed 0.5

# 检查当前参考、整数 PD 和回归。
python scripts/play_motions.py --check
python scripts/check_xq180_reference_pd.py
python -m unittest discover -s tests -v
```

新 AMP 日志目录为 `logs/rsl_rl/xq180_amp_locomotion_no_recovery_userpd_v5_ampfix_v6/`。普通速度任务因共享新 PD，实验目录同步改为 `xq180_velocity_userpd_v5`。已有进程缓存了旧配置，需要重启；本次没有自动启动大规模 GPU 训练。

AMP 默认训练上限为 100000 轮迭代，每轮采样 24 个控制步；`--agent.max-iterations` 指迭代次数。

部署需与新模型配套 PD、scale 和动作裁剪；ONNX 输出仍是原始网络动作，控制器在生成位置目标及保存上一动作前裁剪到 `[-3,3]`，见 [部署示例](xq180_deployment.md)。更换 PD 后，旧 checkpoint 即使输入输出维度一致也不能直接作为新控制器使用。

历史崩溃还包含已确认的 ONNX 导出冻结归一化器问题；该修复继续保留。移除爬起训练符合当前机器人任务，但不能据此确定历史数值异常的全部原因。

## 验证

v4 阶段 28 项回归检查通过，包含实际髋关节链顺序、重新计算的整数增益、名称绑定、17 段专家目录、训练/回放均关闭恢复、tile 高度奖励和速度课程。播放器通过当前 7417 帧的无窗口 FK 检查。CPU 实际 AMP Flat/Rough 环境均确认只加载 7417 个行走参考帧、没有恢复帧和延迟终止，随机重置保持直立，人工降低根部后在一个控制步内终止。Flat 另外通过一次真实 PPO/AMP 更新，使用 17 段专家数据，动作裁剪 3、固定学习率 0.0003，更新后参数有限。上述短检查不代表长期收敛验证。
