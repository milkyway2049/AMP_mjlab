# XQ180 神经网络部署文档

当前 AMP 训练使用 `ampfix_v6`，修复归一化与梯度处理，并增加分阶段学习率及 KL 停止。旧 checkpoint 的训练迁移见 [修复与续训说明](xq180_amp_fix.md)。

本文说明如何将本仓库的 **`XQ180-AMP-Flat`** 策略接入 ONNX 推理和机器人控制循环。机器人为 19 自由度，单机器人输入 `[1,264]`、输出 `[1,19]`。模型与动作适配见 [XQ180 说明](xq180.md)，程序可读参数见 [xq180_policy_io.json](xq180_policy_io.json)。

## 1. 参数版本与对应模型

当前训练只使用 `src/assets/motions/xq180/amp_ik/WalkandRun/` 的 17 段动作，移除恢复重置和延迟终止。AMP 实验目录为 `xq180_amp_locomotion_no_recovery_userpd_v5_ampfix_v6`。

当前 PD 为 **`user_pd_v5`**，采用用户指定的整数表，shoulder pitch 为 50/2，见 [仅行走训练说明](xq180_locomotion.md)。PD 与对应 action scale 已更新，需重新训练并导出。此前 G1 构型推导版本保存在 [v4 参数快照](xq180_policy_io_g1_chain_v4.json)。

本地已有的 `logs/rsl_rl/xq180_amp_locomotion/2026-10-19_14-04-44/policy.onnx` 来自旧 PD 和旧 scale。该模型此前通过 ONNX checker、metadata 核对和一次 ReferenceEvaluator 前向检查，参数保存在 [旧模型接口 JSON](xq180_policy_io_legacy.json)。旧模型应配套旧参数回放；其相同的 `[264] → [19]` shape 不能证明与新参数兼容。新参数 JSON 不再附带旧模型的核验标记。

部署时复制所选 ONNX，并一同保存匹配的 `params/env.yaml`、`params/agent.yaml`、接口 JSON 和 hash。训练保存会覆盖 run 中的 `policy.onnx`。尚未完成新参数策略训练、PyTorch/ONNX 数值对齐、ONNX Runtime 闭环或实机验证。

| 项目 | XQ180-AMP-Flat |
|---|---|
| 输入 | `obs`，float32，ONNX 声明 `[batch,264]`，单机器人用 `[1,264]` |
| 输出 | `actions`，float32，ONNX 声明 `[batch,19]`，单机器人用 `[1,19]` |
| Actor | `264 → 512 → 256 → 128 → 19`；隐藏层 ELU，输出线性 |
| 策略周期 | 20 ms，50 Hz；训练仿真步长 5 ms，decimation 4 |
| 历史 | 单帧 66 维，4 帧，按时间从最旧到最新展平 |
| 归一化 | 默认 AMP 导出包含训练得到的观测归一化；控制器输入原始物理观测 |
| 缩放与裁剪 | Flat actor 六项额外 scale 均为 1；网络输出无 tanh，应用动作裁剪为 `[-3,3]`，观测不额外裁剪 |

Critic 输入为 672 维，AMP 判别器每帧为 165 维，均用于训练。部署 Actor 只需下面列出的 264 维输入，在线不需要参考动作、身体 FK 观测或机器人 meshes。

## 2. 观测、坐标和时序

单帧排列为：

```text
frame_t = [omega_B(3), gravity_B(3), command(3),
           q - q0(19), dq - dq0(19), previous_action(19)]
obs_t = [frame_(t-3), frame_(t-2), frame_(t-1), frame_t]
```

| 单帧切片（右端不包含） | 内容与单位 |
|---|---|
| `[0:3]` | `base_link` 坐标系的 IMU 角速度，rad/s |
| `[3:6]` | `R_WB.T @ [0,0,-1]`，单位重力方向 |
| `[6:9]` | 机体系目标速度 `[vx,vy,wz]`，m/s、m/s、rad/s |
| `[9:28]` | 19 个关节的 `q-q0`，rad |
| `[28:47]` | 19 个关节的速度，rad/s；本配置 `dq0` 全为 0 |
| `[47:66]` | 上一次执行动作，已裁剪到 `[-3,3]`，尚未乘 action scale、尚未加 q0 |

四个历史块在最终输入中的切片为 `[0:66]`、`[66:132]`、`[132:198]`、`[198:264]`。首次进入策略时上一动作归零，用当前首帧重复填满四个槽位；退出后重新进入策略，也要重新初始化这两个状态。每 20 ms 采样一次状态、构造一帧、更新历史并推理，底层更高频的电机循环保持本周期的目标。

### 2.1 IMU 与重力坐标

机体轴为 **x 前、y 左、z 上**。训练 IMU site 名为 `imu_in_pelvis`，实际位于 XQ180 的 `base_link` 原点，方向与 `base_link` 一致。`waist_Link` 是 AMP 身体观测的 anchor，actor 的 gyro 和重力仍以 `base_link` 为参考。

`R_WB` 表示从 `base_link` 到世界的旋转。机体直立时重力输入约为 `[0,0,-1]`。MuJoCo 四元数为 wxyz；若硬件使用 xyzw，应先转换顺序、归一化，并确认旋转方向。IMU 的安装旋转和角速度单位需转换为本模型定义；若安装在腰部以上，还应结合腰关节状态推导根部姿态和角速度。单位重力方向来自姿态，不能直接填入以 m/s² 为单位的原始加速度。

训练的 encoder bias 和观测噪声用于随机化，部署时不额外注入。此 Flat actor 不使用额外的 `dq × 0.05`、`omega × 0.25` 等缩放，也不重复执行 ONNX 内已有的归一化。

### 2.2 速度命令和训练范围

正 `vx` 向前，正 `vy` 向左，正 `wz` 绕 +z 转动。网络只接收最终三个速度目标，不包含 heading 角或 gait phase。

| 采样阶段 | vx (m/s) | vy (m/s) | wz (rad/s) |
|---|---|---|---|
| 初始、前 2000 轮、默认回放 | `[-0.3,0.8]` | `[-0.2,0.2]` | `[-0.5,0.5]` |
| 计数 `>48000`，约 2000 轮后 | `[-0.5,1.2]` | `[-0.3,0.3]` | `[-0.75,0.75]` |
| 计数 `>144000`，约 6000 轮后 | `[-0.8,1.8]` | `[-0.5,0.5]` | `[-1,1]` |

课程在重置时按 `common_step_counter` 更新，计数不乘环境数量，按每轮 24 步换算；命令随后重采样生效。训练每 4–8 秒重采样，10% 为站立命令，25% 使用 heading 控制，航向范围 `[-π/2,π/2]`。默认回放使用初始范围，评估更高速度时应显式修改命令范围。这些范围不是实机性能保证。

## 3. 关节顺序、PD 和动作解码

顺序为 **左腿 5 → 右腿 5 → 腰 1 → 左臂 4 → 右臂 4**。髋关节顺序为 yaw、roll、pitch，右腿从索引 5 开始，腰为 10，右臂从 15 开始。XQ180 的踝关节只有 pitch；名称 `waist_point`、`leftelbow_joint` 和 `rightelbow_joint` 必须按原名处理。

反馈 q/dq、模型输出、q0、Kp、Kd 和 action scale 全部按下表排列。硬件 SDK 若采用其他电机顺序，应按名称建立读取和发送两个方向的映射，并统一零位、方向、弧度单位及减速比。

PD 采用用户指定整数表，左右相同，armature 为 0.01。Kp 单位 Nm/rad，Kd 单位 Nm·s/rad。

| 索引 | 关节 | q0 (rad) | Kp | Kd | action scale | URDF 力矩上限 (Nm) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `left_hip_yaw_joint` | 0 | 200 | 10 | 0.187500000 | 150 |
| 1 | `left_hip_roll_joint` | 0 | 400 | 20 | 0.259375000 | 415 |
| 2 | `left_hip_pitch_joint` | -0.25 | 300 | 15 | 0.345833333 | 415 |
| 3 | `left_knee_joint` | 0.5 | 300 | 15 | 0.345833333 | 415 |
| 4 | `left_ankle_joint` | -0.25 | 100 | 5 | 0.438577314 | 415 |
| 5 | `right_hip_yaw_joint` | 0 | 200 | 10 | 0.187500000 | 150 |
| 6 | `right_hip_roll_joint` | 0 | 400 | 20 | 0.259375000 | 415 |
| 7 | `right_hip_pitch_joint` | -0.25 | 300 | 15 | 0.345833333 | 415 |
| 8 | `right_knee_joint` | 0.5 | 300 | 15 | 0.345833333 | 415 |
| 9 | `right_ankle_joint` | -0.25 | 100 | 5 | 0.438577314 | 415 |
| 10 | `waist_point` | 0 | 100 | 5 | 0.375000000 | 150 |
| 11 | `left_shoulder_pitch_joint` | -0.1 | 50 | 2 | 0.180000000 | 36 |
| 12 | `left_shoulder_roll_joint` | 0.15 | 50 | 2 | 0.180000000 | 36 |
| 13 | `left_shoulder_yaw_joint` | 0 | 25 | 1 | 0.360000000 | 36 |
| 14 | `leftelbow_joint` | -0.5 | 25 | 1 | 0.360000000 | 36 |
| 15 | `right_shoulder_pitch_joint` | -0.1 | 50 | 2 | 0.180000000 | 36 |
| 16 | `right_shoulder_roll_joint` | -0.15 | 50 | 2 | 0.180000000 | 36 |
| 17 | `right_shoulder_yaw_joint` | 0 | 25 | 1 | 0.360000000 | 36 |
| 18 | `rightelbow_joint` | -0.5 | 25 | 1 | 0.360000000 | 36 |

```text
g1_scale_i = 0.25 * G1_effort_limit_i / G1_Kp_i
action_scale_i = min(0.25 * XQ180_effort_limit_i / XQ180_Kp_i, g1_scale_i)
applied_action_i = clip(action_i, -3, 3)
q_des_i = q0_i + action_scale_i * applied_action_i
dq_des_i = 0
tau_PD_i = Kp_i * (q_des_i - q_i) - Kd_i * dq_i
```

位置/混合控制接口可发送 q_des、零目标速度、上述 Kp/Kd 和零前馈力矩。使用纯力矩接口时，在上层计算 PD 并按硬件力矩约束处理，避免重复叠加驱动器内部 PD。两次策略推理之间保持 q_des，底层 PD 可在更高频率执行。

模型未自动将 q_des 裁剪到关节范围；训练的 `soft_joint_pos_limit_factor=0.9` 是奖励中使用的软限位，不会自动裁剪策略目标。URDF 位置限位及速度额定值见 JSON，左右 hip yaw/roll 和 shoulder roll 的限位不对称。URDF effort 不是已验证的持续电机力矩，最终应遵守实际驱动器约束。引入额外动作滤波或裁剪会改变闭环行为，应在对应仿真中评估。

**精度注意：** ONNX metadata 的 `action_scale` 仅保留三位小数，不能用其舍入值重建完整精度数组。踝动作缩放沿用 G1 角度幅度上限，当前为约 0.438577 rad/单位动作。部署参数使用本文表格、接口 JSON 或匹配 run 的 `env.yaml` 中的完整数值，metadata 用于核对名称和参数来源。

肘角减 π/2 是 G1 动作离线转换到 XQ180 时的坐标补偿，已用于参考动作生成。部署时直接使用 XQ180 URDF 定义下的实测关节角；不再对 q0 或网络输出重复减 π/2。电机原始编码器到该关节角的零位标定由硬件映射负责。

仿真 armature 为每关节 0.01 kg·m²，是初始估计，不是电机命令字段。模型左肩 pitch 的原始 CAD 惯量在 MJCF 中做过三角不等式修正，详情见 [适配说明](xq180.md)；该模型及 PD 尚未完成实机辨识。

## 4. 导出和模型交付

所有命令在仓库根目录、mjlab 环境中执行：

```bash
conda activate mjlab
python scripts/train.py XQ180-AMP-Flat --env.scene.num-envs=4096
python scripts/play.py XQ180-AMP-Flat \
  --checkpoint-file /path/to/xq180_run/model_XXXX.pt --num-envs 1
```

训练保存时 run 目录生成 `policy.onnx`，后续保存会覆盖。指定 checkpoint 回放默认导出到该 run 的 `export/XQ180-AMP-Flat_model_XXXX.onnx`；终端应打印成功导出路径。使用具体 checkpoint 可让交付文件与训练迭代对应。

本地旧参数模型路径（配套 legacy JSON）：

```text
logs/rsl_rl/xq180_amp_locomotion/2026-10-19_14-04-44/policy.onnx
```

AMP 导出器包含观测归一化和确定性 actor，不采样高斯探索噪声。每次换模型时读取真实输入输出、dtype 和 shape，并检查模型中是否包含归一化。训练保存附带 joint_names、PD、默认角和 scale metadata；回放的独立导出路径不保证包含同样的 metadata，因此交付时应携带匹配的配置文件。

## 5. ONNX Runtime 控制循环示例

以下示例在 CPU 使用 ONNX Runtime，读取 JSON 中的完整精度参数，维护四帧历史和上一动作。q/dq 已转换到策略关节顺序，omega_B 和 R_WB 已转换到 `base_link` 定义。代码输出关节位置目标，由机器人控制器接入驱动器。metadata 存在时会核对参数以发现新旧配置混用；独立导出缺少 metadata 时，需根据对应 run 配置核对。

```python
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class XQ180Policy:
    def __init__(self, model_path, config_path):
        cfg = json.loads(Path(config_path).read_text())
        self.names = [j["name"] for j in cfg["joints"]]
        self.q0 = np.array([j["default_position"] for j in cfg["joints"]], np.float32)
        self.scale = np.array([j["action_scale"] for j in cfg["joints"]], np.float32)
        self.kp = np.array([j["kp"] for j in cfg["joints"]], np.float32)
        self.kd = np.array([j["kd"] for j in cfg["joints"]], np.float32)
        self.action_clip = cfg["action_clip"]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp, = self.session.get_inputs()
        out, = self.session.get_outputs()
        assert (inp.name, inp.type, inp.shape[-1]) == ("obs", "tensor(float)", 264)
        assert (out.name, out.type, out.shape[-1]) == ("actions", "tensor(float)", 19)
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "joint_names" in metadata and metadata["joint_names"].split(",") != self.names:
            raise ValueError("Model joint order differs from the parameter JSON")
        for key, values in (("joint_stiffness", self.kp), ("joint_damping", self.kd),
                            ("action_scale", self.scale), ("default_joint_pos", self.q0)):
            if key in metadata:
                actual = np.fromstring(metadata[key], sep=",")
                if actual.shape != (19,) or not np.allclose(actual, values, atol=0.00051, rtol=0):
                    raise ValueError(f"Model {key} differs from the parameter JSON")
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(19, np.float32)
        self.history = None

    def step(self, q, dq, omega_B, R_WB, command):
        q = np.asarray(q, dtype=np.float32).reshape(19)
        dq = np.asarray(dq, dtype=np.float32).reshape(19)
        omega_B = np.asarray(omega_B, dtype=np.float32).reshape(3)
        R_WB = np.asarray(R_WB, dtype=np.float32).reshape(3, 3)
        command = np.asarray(command, dtype=np.float32).reshape(3)
        gravity_B = R_WB.T @ np.array([0, 0, -1], np.float32)
        frame = np.concatenate((omega_B, gravity_B, command,
                                q - self.q0, dq, self.previous_action))
        if not np.isfinite(frame).all():
            raise ValueError("Non-finite robot observation")
        if self.history is None:
            self.history = np.repeat(frame[None, :], 4, axis=0)
        else:
            self.history = np.concatenate((self.history[1:], frame[None, :]), axis=0)
        obs = np.ascontiguousarray(self.history, dtype=np.float32).reshape(1, 264)
        action = self.session.run(["actions"], {"obs": obs})[0]
        if action.shape != (1, 19) or not np.isfinite(action).all():
            raise ValueError("Invalid policy action")
        self.previous_action = np.clip(action[0], -self.action_clip, self.action_clip).copy()
        return self.q0 + self.scale * self.previous_action


policy = XQ180Policy("policy.onnx", "docs/xq180_policy_io.json")
# 首次调用的构造输入；实机应替换为同期采样的反馈和命令。
q_des = policy.step(policy.q0, np.zeros(19), np.zeros(3), np.eye(3), np.zeros(3))
# 随后每 20 ms 调用一次 step；退出/重新进入策略时调用 reset。
```

C++ ORT 使用相同的连续 264 个 float，输入 shape `{1,264}`，输出 19 个 float；保持相同的观测排列、历史更新和动作解码。若 SDK 在输出后还改变动作，上一动作的处理也应与所采用的训练/仿真接口对齐。

部署验证先比较同一组仿真观测下 PyTorch 与 ONNX 的输出，再检查 ONNX 闭环仿真和硬件关节映射。状态无效或推理失败时由已有控制状态管理接管。本文确认的是接口与一次参考前向计算，不代表已完成运动性能验收。

## 6. 其他任务和源码入口

| Task | 单帧 actor | 历史 | 输入维度 | 输出维度 |
|---|---:|---:|---:|---:|
| XQ180-AMP-Flat | 66 | 4 | 264 | 19 |
| XQ180-AMP-Rough | 253 | 4 | 1012 | 19 |
| XQ180-Flat | 68 | 1 | 68 | 19 |
| XQ180-Rough | 255 | 1 | 255 | 19 |

普通速度任务在 command 后增加两维 phase `[sin(2πφ),cos(2πφ)]`，周期 0.6 s，命令范数小于 0.1 时置零。Rough 在每帧末尾增加 187 个高度扫描值，17×11 网格，间隔 0.1 m，以 `base_link` 为参考、按 yaw 对齐，x 在每行内变化。数值为 `(sensor_z-hit_z)/5`，未命中为 1。其他任务应按各自实际导出模型构造输入，普通速度任务的 ONNX tensor 名称也应读取模型确认。

- [XQ180 机器人配置](../src/assets/robots/xq180/xq180_constants.py)：默认站姿、PD、力矩和动作缩放。
- [AMP 基础环境](../src/tasks/amp_loco/amp_env_cfg.py)：观测顺序、历史、策略周期和命令课程。
- [XQ180 AMP 配置](../src/tasks/amp_loco/config/xq180/env_cfgs.py)：Flat/Rough 差异和传感器。
- [AMP ONNX 导出器](../src/tasks/amp_loco/rl/runner.py)：观测归一化封装与 metadata。
- [回放和 checkpoint 导出](../scripts/play.py)：导出入口与路径。
- [历史观测补丁](../mjlab_patch/mjlab/managers/observation_manager.py)：按整帧时间顺序展平。
- [接口参数 JSON](xq180_policy_io.json)：19 关节完整参数及模型核验记录。
