# Hunter180 神经网络部署文档

本文对应本仓库 `Hunter180-AMP-Flat` 的 23 关节配置。模型适配和训练入口见 [Hunter180 说明](hunter180.md)，程序可读参数见 [hunter180_policy_io.json](hunter180_policy_io.json)。这是训练配置的接口说明；尚未核验已训练的 Hunter180 ONNX 或实机效果。

## 1. 推理接口

| 项目 | AMP-Flat 配置 |
|---|---|
| 输入 | AMP 导出器使用 `obs`，float32，单机器人 `[1, 312]` |
| 输出 | `actions`，float32，`[1, 23]` |
| Actor | `312 → 512 → 256 → 128 → 23`，隐藏层 ELU，输出线性 |
| 控制 | 50 Hz，每 20 ms 一次策略推理；仿真步长 5 ms、decimation 4 |
| 历史 | 单帧 78 维，4 帧，从最旧到最新，按整帧展平 |
| 归一化 | 默认训练启用；AMP ONNX 导出器将观测归一化包含在模型内 |
| 裁剪 | 当前没有额外观测 scale/clip，也没有 action clip 或输出 tanh |

Critic 为训练使用的 720 维观测，AMP 判别器使用 165 维观测；实机 Actor 不需要输入它们，也不在线加载参考动作或 meshes。Hunter130 的 `[1,336] → [1,25]` 模型不能用于 Hunter180。

## 2. 观测、坐标与历史

```text
frame_t = [omega_b(3), gravity_b(3), command(3),
           q-q0(23), dq(23), previous_action(23)]
obs_t = [frame_(t-3), frame_(t-2), frame_(t-1), frame_t]
```

| 切片（右端不包含） | 内容 |
|---|---|
| `[0:3]` | pelvis 坐标系 gyro，rad/s |
| `[3:6]` | `R_WB.T @ [0,0,-1]`，单位重力方向 |
| `[6:9]` | `[vx,vy,wz]`，单位 m/s、m/s、rad/s |
| `[9:32]` | 23 关节位置减本模型默认姿态，rad |
| `[32:55]` | 23 关节速度，rad/s；默认速度全为零 |
| `[55:78]` | 上一次策略周期的动作，未乘 scale、未加 q0 |

机体为 x 前、y 左、z 上。MuJoCo 四元数为 wxyz，ROS 通常为 xyzw；确认表示的是机体到世界的旋转，并核对 IMU 安装转换。直立的重力输入应约为 `[0,0,-1]`，不能使用 `[0,0,-9.81]`。如果传感器不在 pelvis，应补偿安装/腰部运动后再构造输入。

首次进入策略时上一动作归零，用当前第一帧重复填满四个槽位。每 20 ms 读取状态、更新一次历史、推理得到动作、计算目标并保存动作；底层高速循环不要每次追加历史。不得重复做 ONNX 已包含的观测归一化，或沿用其他项目的速度乘 0.05 等缩放。

## 3. 关节顺序、指定 3 倍的 PD 和动作

左腿 6 → 右腿 6 → 腰 3 → 左臂 4 → 右臂 4，右臂从索引 19 开始。反馈、输出、q0、Kp/Kd/scale 必须使用同一顺序；SDK 若是交错电机顺序，应按关节名建立双向映射。

整机质量比为 `64.34554445 / 30.87757522 ≈ 2.0839`；考虑重量并按用户指定，最终使用 **3 倍**：`Kp_180=3*Kp_130`、`Kd_180=3*Kd_130`。以下表格为舍入值，JSON 保存完整精度。Kp 单位 Nm/rad，Kd 单位 Nm·s/rad。

| 索引 | 关节 | q0 (rad) | Kp | Kd | scale (rad) | 力矩上限 (Nm) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `left_hip_pitch_joint` | -0.25 | 297.295283 | 18.926406 | 0.378411654 | 450 |
| 1 | `left_hip_roll_joint` | 0.00 | 297.295283 | 18.926406 | 0.378411654 | 450 |
| 2 | `left_hip_yaw_joint` | 0.00 | 120.537716 | 7.673669 | 0.933317835 | 450 |
| 3 | `left_knee_joint` | 0.50 | 297.295283 | 18.926406 | 0.378411654 | 450 |
| 4 | `left_ankle_pitch_joint` | -0.25 | 85.503739 | 5.443337 | 1.818633928 | 622 |
| 5 | `left_ankle_roll_joint` | 0.00 | 85.503739 | 5.443337 | 0.602312844 | 206 |
| 6 | `right_hip_pitch_joint` | -0.25 | 297.295283 | 18.926406 | 0.378411654 | 450 |
| 7 | `right_hip_roll_joint` | 0.00 | 297.295283 | 18.926406 | 0.378411654 | 450 |
| 8 | `right_hip_yaw_joint` | 0.00 | 120.537716 | 7.673669 | 0.933317835 | 450 |
| 9 | `right_knee_joint` | 0.50 | 297.295283 | 18.926406 | 0.378411654 | 450 |
| 10 | `right_ankle_pitch_joint` | -0.25 | 85.503739 | 5.443337 | 1.818633928 | 622 |
| 11 | `right_ankle_roll_joint` | 0.00 | 85.503739 | 5.443337 | 0.602312844 | 206 |
| 12 | `waist_yaw_joint` | 0.00 | 120.537716 | 7.673669 | 0.829615853 | 400 |
| 13 | `waist_roll_joint` | 0.00 | 85.503739 | 5.443337 | 0.760200677 | 260 |
| 14 | `waist_pitch_joint` | 0.00 | 85.503739 | 5.443337 | 1.040890158 | 356 |
| 15 | `left_shoulder_pitch_joint` | -0.10 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 16 | `left_shoulder_roll_joint` | 0.20 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 17 | `left_shoulder_yaw_joint` | 0.00 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 18 | `left_elbow_joint` | -0.50 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 19 | `right_shoulder_pitch_joint` | -0.10 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 20 | `right_shoulder_roll_joint` | -0.20 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 21 | `right_shoulder_yaw_joint` | 0.00 | 42.751869 | 2.721669 | 0.935631603 | 160 |
| 22 | `right_elbow_joint` | -0.50 | 42.751869 | 2.721669 | 0.935631603 | 160 |

动作解码：

```text
scale_i = 0.25 * URDF_effort_limit_i / Kp_i
q_des_i = q0_i + scale_i * action_i
tau_i = Kp_i * (q_des_i - q_i) - Kd_i * dq_i
```

当前 scale 已按 3 倍的 Kp 重算，不能统一使用 0.25，也不能沿用 Hunter130 的完整数组。位置/混合控制可传 q_des、零目标速度、Kp/Kd 和零前馈力矩；力矩控制则由上层计算 PD，避免与电机内部 PD 叠加。两次策略更新之间保持目标，实际电机侧需核对零位、方向、单位、减速比及真实限位。

3 倍 PD 是考虑重量后的指定仿真初值，未求解连杆长度变化和各关节的精确等效惯量。URDF 力矩值不是实机持续力矩；SDK 的机械/速度/力矩限制仍需要遵守。接口 JSON 的位置限位来自 URDF，训练 soft limit 0.9 是奖励约束，不会自动裁剪网络的位置目标。

## 4. 训练命令范围

| 阶段 | vx (m/s) | vy (m/s) | wz (rad/s) |
|---|---|---|---|
| 基础/初次采样、回放 | `[-1.5,3]` | `[-1,1]` | `[-1.57,1.57]` |
| 早期课程 | `[-0.5,1]` | `[-0.5,0.5]` | `[-1,1]` |
| 后期课程 | `[-1,2]` | `[-1,1]` | `[-1,1]` |

课程在环境重置时应用，采用 `common_step_counter > 0` 和 `> 120000` 的严格比较。默认每 iteration 24 步，第二阶段约在 5000 个 iteration 后；计数不乘并行环境数量。第二阶段没有再次设置 wz，所以保留 `[-1,1]`。命令采样配置变化不立即重写已有线速度，新的范围用于后续重置/重采样；回放关闭课程，使用较宽的基础范围。

重采样间隔随机 3–8 s；约 5% 为站立命令全零，约 25% 使用 heading。目标世界航向范围 `[-π/2,π/2]`，`wz=clip(0.5*wrap_to_pi(target_heading-current_heading), 当前 wz 范围)`。输入网络的是最终 wz，不能增加 heading 字段。实际部署范围应以所选 run 的训练覆盖和评估结果为准。

## 5. 导出与 ONNX Runtime 对接

```bash
conda activate mjlab
python scripts/train.py Hunter180-AMP-Flat --env.scene.num-envs=4096
python scripts/play.py Hunter180-AMP-Flat \
  --checkpoint-file /path/to/hunter180_run/model_XXXX.pt --num-envs 1
```

训练保存时 run 目录生成 `policy.onnx`，后续保存会覆盖它。指定 checkpoint 回放默认导出至该 run 的 `export/Hunter180-AMP-Flat_model_XXXX.onnx`。明确提供 checkpoint，不依赖本地最新模型自动选择。保存模型副本、SHA256 和对应 `params/env.yaml`、`params/agent.yaml`；修改过配置的 run 需要重新核对接口 JSON。

用实际 ONNX 读取输入/输出名称、dtype、shape；AMP 训练导出通常有 joint_names/PD/scale metadata，而 `play.py` 的独立导出可能没有这些字段。使用 metadata 或匹配的 run 配置核对全部 23 关节参数。

Python ORT 推理核心：

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])
assert session.get_inputs()[0].name == "obs"
assert session.get_inputs()[0].shape[-1] == 312
assert session.get_outputs()[0].name == "actions"
assert session.get_outputs()[0].shape[-1] == 23
# history: (4,78)，已按旧到新构造，首次重复首帧。
obs = np.ascontiguousarray(history, dtype=np.float32).reshape(1,312)
actions = session.run(["actions"], {"obs": obs})[0]
assert actions.shape == (1,23) and np.isfinite(actions).all()
q_des = q0 + action_scale * actions[0]
```

上例的 history、q0、action_scale 由控制器和接口 JSON 提供，不会发送电机命令。C++ ORT 对应连续 312 个 float 和 shape `{1,312}`，输出 23 个 float；推理与状态采样应在 20 ms 周期内完成。

验收时先用同一仿真观测比较 PyTorch 与 ONNX，再做 ONNX 闭环仿真和硬件映射核对。推理异常/无效状态时由上层控制状态管理接管，不继续发送未检查结果。当前文档和 JSON 不代表已完成 Hunter180 网络导出或实机部署。

## 6. 其他任务

| Task | 单帧 actor | 历史 | 输入 | 动作 |
|---|---:|---:|---:|---:|
| Hunter180-AMP-Flat | 78 | 4 | 312 | 23 |
| Hunter180-AMP-Rough | 265 | 4 | 1060 | 23 |
| Hunter180-Flat | 80 | 1 | 80 | 23 |
| Hunter180-Rough | 267 | 1 | 267 | 23 |

普通速度任务在 command 后增加两维 phase，周期 0.6 s，命令范数小于 0.1 时置零。Rough 在每帧末尾增加 187 个 pelvis/yaw 对齐的高度扫描值，17×11 网格、间隔 0.1 m，x 在每行内变化，数值为 `(sensor_z-hit_z)/5`，未命中为 1。上述任务要使用各自实际导出模型和观测布局，不能互换权重。
