# Hunter130 健身垫训练调整

本次修改适用于 `Hunter130-Flat/Rough` 和 `Hunter130-AMP-Flat/Rough`。当前仓库没有可读取的 Hunter130 run 配置或 checkpoint，因此下面的审计针对源码默认配置；不能据此确认正在部署的旧模型训练时用了哪些参数。

## 原训练是否有随机化和噪声

有。训练 actor 开启 `enable_corruption=True`；critic 和 AMP 判别器观测不加噪声。

| 观测项 | AMP 均匀噪声范围 | 普通速度任务 |
|---|---|---|
| IMU 角速度 | ±0.2 rad/s | 相同 |
| 投影重力每个分量 | ±0.05 | 相同 |
| 关节位置 | ±0.01 rad | 相同 |
| 关节速度 | ±0.5 rad/s | ±1.5 rad/s |

原有启动随机化包括：双脚共享摩擦系数 `0.3–1.2`、编码器偏置 `±0.015 rad`、torso 质心 XY 各 `±0.025 m` / Z `±0.03 m`。AMP 从随机参考动作帧重置；普通速度任务随机根节点 XY 和 yaw，但原关节重置偏差为零。原推扰每 `1–3 s` 一次，包括竖直方向和 roll/pitch。Flat 不输入高度扫描，因此高度扫描的噪声不影响 Flat actor。

原配置只随机摩擦，没有随机接触柔顺性。站立命令只占 5%，而 `foot_slip` 在命令接近零时被关掉；AMP 的动作变化惩罚也较弱。这些是针对站立晃动和软垫适应不足需要补足的训练项，尚不是实机根因的测量结论。

## 已实现的调整

实现集中在 [hunter130_robustness.py](../src/tasks/hunter130_robustness.py)，由 Hunter130 的两类环境配置调用。

| 项目 | 新配置 |
|---|---|
| 支撑面 | 每个训练环境启动时以 40% 概率保留默认硬地接触，以 60% 概率使用软接触；双脚共享一次采样 |
| 软接触参数 | `solref=(U[0.03,0.06] s, U[0.9,1.2])`，`solimp=(0.7,0.95,0.01,0.5,2)` |
| PD 随机化 | 训练启动时，Kp 为指定名义值的 `0.9–1.1`，Kd 为 `0.85–1.15`；相对编译默认值采样 |
| 指令延迟 | 训练每次 reset 采样 `0/5/10 ms`，同一环境全部关节共享，整回合保持 |
| 站立命令比例 | 25% |
| 推扰 | 每 `4–8 s`，仅 X `±0.3 m/s`、Y `±0.25 m/s`、yaw `±0.35 rad/s` |
| 脚滑 | 站立及行走均惩罚接触脚的 XY 速度平方，权重 `-0.5` |
| 站立 | 惩罚相对默认站姿偏移及关节运动；运动命令下关闭该项 |
| 平滑 | 动作一阶变化平方惩罚权重 `-0.05` |
| AMP 躯干稳定 | 新增 torso 倾斜及 roll/pitch 角速度惩罚，收紧原 pelvis 稳定奖励的 std 到 `1.0` |
| 普通速度躯干稳定 | 原角速度惩罚权重由 `-0.05` 调至 `-0.1` |

新脚滑、站立和 AMP torso 惩罚在正在延迟终止的恢复环境中关闭；AMP 原有 40% 恢复环境、250 步恢复窗口保持原配置。

足部 geom 的接触优先级高于地面，因此本实现修改足部的 `geom_solref/geom_solimp`。MuJoCo 的接触优先级和求解器参数定义见[官方建模文档](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。这是覆盖软支撑的接触近似，尚未根据你的垫子标定，不包含泡棉真实形变、厚度或滞回模型；参数范围需要通过训练和实测比较验证。

## 名义 PD 与部署

| 关节 | Kp | Kd |
|---|---:|---:|
| 髋 pitch / roll / yaw、膝 | 100 | 8 |
| 踝 pitch / roll | 30 | 5 |
| 肩 pitch / roll / yaw、肘、腕 roll | 25 | 1 |
| 腰 yaw / roll / pitch | 60 | 5 |

[部署 JSON](hunter130_policy_io.json) 已同步。`action_scale = round(0.25 * effort_limit / Kp, 3)` 随 Kp 重算，因此旧策略对应的动作含义也发生变化。应按新参数重训；若微调旧 checkpoint，需要验证其在新 PD/scale 下的闭环表现。旧 ONNX 文件不会因修改本仓库而更新，实机控制器也需要使用新策略对应的参数。

AMP-Flat 仍是 `336 → 25`，4 帧时间历史、50 Hz；普通 Flat 仍是 `86 → 25`。Hunter180 使用冻结的历史 PD 基准，避免被 Hunter130 本次调参连带改变。

## 训练与检查

```bash
conda activate mjlab
python -m unittest discover -s tests -v
python scripts/train.py Hunter130-AMP-Flat \
  --env.scene.num-envs=4096 --agent.run-name=mat_pd_20261028
# 普通速度训练可把任务名改成 Hunter130-Flat。
```

当前 Hunter130 的 `play=True` 使用软接触，关闭 actor 噪声、推扰、训练用 PD 随机化和指令延迟；摩擦、编码器偏置、质心偏移仍按原逻辑随机化。软接触参数也会随机采样。

```bash
python scripts/play.py Hunter130-AMP-Flat \
  --checkpoint-file /path/to/new_run/model_XXXX.pt --num-envs 1
```

做硬地对照时，可在 `configure_hunter130_robustness` 的 play 配置中将 `hard_ground_ratio` 设为 `1.0`；`0.0` 表示全部软接触。`play.py` 当前不支持从命令行覆盖环境参数。比较同一新策略在硬地/软接触下的零速漂移、roll/pitch 角速度、脚滑和摔倒次数，再确认实机健身垫表现。单元测试和有限步仿真只验证实现与数值，不代表已验证步态效果。

本次验证：32 项测试通过；AMP-Flat 训练、普通 Flat 训练和 AMP-Flat play 各 8 个 GPU 环境完成 16 步软接触数值检查，观测/奖励保持有限值，并检查了双脚接触参数一致、训练关节延迟同步及输入维度。25 个关节的部署 JSON PD/scale 与编译后的名义模型一致。本次没有启动完整重训，也未完成新策略的实机验证。
