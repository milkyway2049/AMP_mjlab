# XQ180 参考动作播放与训练崩溃检查

> 本文为历史检查记录。当前采用仅行走、G1 惯量比例整数 PD，参数及命令以 [当前方案](xq180_locomotion.md) 为准。

检查日期：2026-10-21。当前参考库为 `src/assets/motions/xq180/amp_ik`，共 18 段、9992 帧、50 Hz，总时长 199.84 秒。包含 17 段走跑/转向和 1 段跌倒恢复。

## 连续播放所有训练参考动作

在仓库根目录、已激活 `mjlab` 的桌面终端执行：

```bash
python scripts/play_motions.py --speed 0.5
```

这是半速播放，走跑/转向在前，恢复动作在最后；默认循环整库。使用 `--speed 1` 恢复原速，`--once` 播完一轮退出。窗口显示文件名、帧数和时间，终端也打印当前片段。

| 按键 | 操作 |
|---|---|
| 空格 | 暂停/继续 |
| 右/左方向键，或 `]` / `[` | 下一段/上一段 |
| `,` / `.` | 暂停并后退/前进一帧 |
| `R` | 从当前片段开头重播 |
| `F` | 开关根部跟随镜头 |
| Esc | 关闭窗口 |

从某段开始，或对比原 G1 / 旧映射：

```bash
python scripts/play_motions.py --start walk_forward_loop_002__A022 --speed 0.5
python scripts/play_motions.py --robot g1 --speed 0.5
python scripts/play_motions.py --motion-dir src/assets/motions/xq180/amp --speed 0.5

# 无窗口检查全部加载、坐标与 FK。
python scripts/play_motions.py --check
```

播放器直接设置每帧浮动根和关节位置，再执行 FK。它用于检查**专家参考动作**，不运行 PD，也不表示训练后的策略能复现动作。`scripts/play.py --agent zero` 则不属于参考轨迹回放。

重点观察支撑脚是否滑动、落脚是否突变、膝髋是否抖动、身体高度是否突然变化，以及恢复时是否出现缺少合理支撑的姿态。片段切换会直接跳到下一段起始姿态，没有插值拼接；切换跳变不属于单段动作问题。

当前 IK 库通过了软限位、速度、地面间隙和自穿透的全帧运动学检查，详见 [重定向报告](xq180_retarget.md)。近地足底速度代理均值仍约 0.270 m/s，恢复动作根部竖直加速度 P99 约 26.6 m/s²；动作仍有动态可实现性风险，尚未检查接触力、摩擦锥和全身平衡。运动学合格不能证明适合作为稳定策略的专家数据。

## 训练日志证据

检查的 run：`logs/rsl_rl/xq180_amp_locomotion_userpd_ik_v1/2026-10-20_14-22-19/`。其保存的配置确认使用 `amp_ik` 和原用户 PD 表，动作未裁剪。

| 迭代 | 观测到的现象 |
|---:|---|
| 40,000 | 平均回报 34.61，平均 episode 长度 867 步，跳过批次为 0 |
| 48,200 | 平均回报 32.42，episode 长度 836 步，跳过批次为 0 |
| 48,201 | `Loss/skipped_non_finite_batches` 首次大于 0 |
| 48,202 | 每轮 20 个批次全部跳过；显示为 0 的 loss 不代表收敛 |
| 48,207 | 动作变化惩罚从约 -0.06 变为 -88.36，平均回报 -2510.90 |
| 48,239 | 速度误差指标首次出现非有限值 |
| 48,493 | 平均回报首次出现非有限值 |
| 49,000 | 动作惩罚约 -5.42e30，episode 长度仅约 92 步 |
| 97,000 | 仍每轮跳过 20 个更新批次；所检查的策略/判别器权重与 49,000 相同 |

这表明后半程包含数值异常和停止有效学习，不能只解释为“训练还没收敛”。所检查的 40,000、48,200、48,300、49,000、97,000 checkpoint 中，策略、判别器和观测归一化参数均有限。现有日志没有保留第一次异常对应的单个环境状态，尚不能确定最初触发因素，更不能认定由 PD 或某个动作直接造成。后期学习率达到 0.01 出现在崩溃之后，不能倒推为最早原因。

另外确认并修复了一个训练缺陷：AMP 保存 checkpoint 时导出 ONNX，导出器直接对训练中的 policy 和 actor 归一化器调用 `.eval()`。第一次保存之后，actor 归一化统计停止更新；checkpoint 中 actor 的 `count=98304`，恰好是 `4096×24` 个首轮样本，而 critic 的 count 已达到约 1e8。现在导出独立深拷贝，成功或失败都不会改变训练对象的设备、模式或统计。这个缺陷有源码、checkpoint 和回归测试证据，但尚未通过长训练对照证明它单独引起第 48,201 轮异常。

## 本次 PD 调整

当前方案名为 `leg_damping_v2`。保留原 Kp 和 action scale，仅降低 hip roll/pitch、knee、ankle 的阻尼；其 Kd/Kp 从 0.1 s 降到约 0.0667 s，接近 G1 对应组的约 0.06366 s。

| 关节组 | Kp | Kd |
|---|---:|---:|
| hip yaw | 300 | 30 |
| hip roll / pitch、knee | 300 | 20 |
| ankle pitch | 150 | 10 |
| waist yaw | 300 | 30 |
| shoulder / elbow | 50 | 2.5 |

对全部参考执行固定根、关闭重力/接触、5 ms implicitfast、50 Hz 位置目标跟踪，结果如下。误差在每次目标更新时计算，按帧数和 19 个关节聚合 RMS。

| 方案 | 走跑 RMSE (rad) | 恢复 RMSE (rad) |
|---|---:|---:|
| 原用户表 `user_pd_v1` | 0.12311 | 0.15364 |
| 历史整套 `g1_inertia_v1` | 0.13125 | 0.16057 |
| 当前 `leg_damping_v2` | 0.11425 | 0.14107 |

当前方案的平均伺服跟踪误差分别下降约 7.2%、8.2%。这项测试不包含重力、浮动根平衡或真实脚地接触，也没有执行 G1 自定义力矩速度曲线；不能据此承诺更快收敛或更稳定的行走。完整逐段数据见 [reference_servo_comparison.json](xq180_retarget/reference_servo_comparison.json)。复现命令：

```bash
python scripts/check_xq180_reference_pd.py
```

原用户表和原 action scale 保存在 [user_pd_v1 接口快照](xq180_policy_io_user_pd_v1.json)，该阶段参数见 [leg_damping_v2 快照](xq180_policy_io_leg_damping_v2.json)。旧 checkpoint 回放须配套原 PD；参考播放器不受 PD 版本影响。

若开始下一轮训练，使用新目录，便于与失败 run 区分：

```bash
python scripts/train.py XQ180-AMP-Flat --env.scene.num-envs=4096
```

新日志目录为 `logs/rsl_rl/xq180_amp_locomotion_legpd_ik_v2/`。本次没有启动长期训练，也未更换或删除参考片段。

## 验证范围

25 项回归检查通过，包含导出成功/失败后训练归一化统计继续更新；另完成小网络的真实 ONNX 导出、checker 和 ReferenceEvaluator 数值对齐，导出后训练统计继续更新。AMP Flat/Rough 在 CPU 上初始化、重置和 8 步检查通过。播放器对 XQ180 和 G1 各 18 段、9992 帧完成无窗口检查；XQ180 保存身体位置与播放 FK 最大差约 0.70 微米。当前沙箱无法连接 `DISPLAY=:1`，未完成原生窗口交互验证，需在本机桌面终端执行播放命令。
