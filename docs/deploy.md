| Index | Joint | q0 (rad) |
|---:|---|---:|
| 0 | `left_hip_pitch_joint` | -0.1 |
| 1 | `left_hip_roll_joint` | 0 |
| 2 | `left_hip_yaw_joint` | 0 |
| 3 | `left_knee_joint` | 0.3 |
| 4 | `left_ankle_pitch_joint` | -0.2 |
| 5 | `left_ankle_roll_joint` | 0 |
| 6 | `right_hip_pitch_joint` | -0.1 |
| 7 | `right_hip_roll_joint` | 0 |
| 8 | `right_hip_yaw_joint` | 0 |
| 9 | `right_knee_joint` | 0.3 |
| 10 | `right_ankle_pitch_joint` | -0.2 |
| 11 | `right_ankle_roll_joint` | 0 |
| 12 | `waist_yaw_joint` | 0 |
| 13 | `waist_roll_joint` | 0 |
| 14 | `waist_pitch_joint` | 0 |
| 15 | `left_shoulder_pitch_joint` | 0.2 |
| 16 | `left_shoulder_roll_joint` | 0.18 |
| 17 | `left_shoulder_yaw_joint` | 0 |
| 18 | `left_elbow_joint` | 0.6 |
| 19 | `left_wrist_roll_joint` | 0 |
| 20 | `right_shoulder_pitch_joint` | 0.2 |
| 21 | `right_shoulder_roll_joint` | -0.18 |
| 22 | `right_shoulder_yaw_joint` | 0 |
| 23 | `right_elbow_joint` | 0.6 |
| 24 | `right_wrist_roll_joint` | 0 |

| 切片 | 维数 | 内容 | 单位/定义 |
|---|---:|---|---|
| `[0:3]` | 3 | `base_ang_vel` | pelvis 坐标系角速度，rad/s，x/y/z |
| `[3:6]` | 3 | `projected_gravity` | 单位重力方向投影到 pelvis 坐标系 |
| `[6:9]` | 3 | `command` | `[vx, vy, wz]`，m/s、m/s、rad/s |
| `[9:34]` | 25 | `joint_pos` | `q - q0`，rad，按第 5 节顺序 |
| `[34:59]` | 25 | `joint_vel` | `dq - dq0`，rad/s；当前 `dq0` 全为 0 |
| `[59:84]` | 25 | `actions` | 上一次策略周期交给 action manager 的动作，尚未乘 scale/加 q0 |

| 项目 |  |
|---|---|
| 输入 | `obs`，`float32`，实机单机器人使用 `[1, 336]` |
| 输出 | `actions`，`float32`，`[1, 25]` |
| 策略周期 | 20 ms，即 50 Hz |
| 单帧观测 | 84 维 |
| 历史 | 4 帧，按时间展开，从最旧到最新 |