# Hunter180 适配说明

将 `/home/milkyway/opensource/hunter130_train/legged_lab/assets/hunter180` 中的 URDF 和 meshes 接入 AMP_mjlab，机器人命名为 `hunter180`。模型资源位于 `src/assets/robots/hunter180/`；所有命令在仓库根目录、`conda activate mjlab` 环境执行。

## 机器人与模型

- 23 个活动关节：左腿 6、右腿 6、腰部 3、左臂 4、右臂 4，没有腕部活动关节；24 个连杆，总质量 64.34554445 kg。
- [hunter180.urdf](../src/assets/robots/hunter180/hunter180.urdf) 保留原关节结构、坐标、惯量、限位及力矩/速度参数，mesh 路径规范为本地 `meshes/`。复制 URDF 实际引用的网格，运行不依赖外部原目录。
- [hunter180.xml](../src/assets/robots/hunter180/xmls/hunter180.xml) 增加浮动基座、pelvis IMU、足底 sites 和角动量传感器。动力学使用 URDF 完整惯量，不从视觉碰撞网格推算质量。
- 足部视觉和碰撞均使用原始 `ankle_roll_link.STL`，不替换成盒体/胶囊、不减面。MuJoCo mesh 碰撞默认使用凸包。其他部位采用从 STL 范围拟合的盒体/胶囊，并排除关节链距离不超过两个关节的连杆碰撞。
- 初始站姿沿用来源模型：hip pitch `-0.25`、knee `0.5`、ankle pitch `-0.25`、shoulder pitch `-0.1`、elbow `-0.5`，左右 shoulder roll 为 `+0.2/-0.2` rad；其余为零。pelvis 高度 0.925 m，名义姿态双脚 sites 离地约 8.6 mm，无初始自碰撞。

## 按重量参考与指定倍数设置 PD

以本项目 Hunter130 在 2026-10-16 的对应关节 PD 为固定基准（Hunter130 后续调参不会改变 Hunter180）。两台机器人的质量比约为 2.0839，当前按用户指定的 **3 倍**设置 Hunter180 的 Kp 和 Kd：

```text
mass_ratio = Hunter180_mass / Hunter130_mass
           = 64.34554445 / 30.87757522
           ≈ 2.0839

PD_scale = 3.0
Kp_180 = Kp_130 * PD_scale
Kd_180 = Kd_130 * PD_scale
```

这一历史 Hunter130 基准 PD 来自项目 G1 对应关节的小幅取整值。Hunter180 对应组的 Kp/Kd 为：腿部主关节 `300/18.9`，hip yaw 和 waist yaw `120/7.8`，ankles 和 waist pitch/roll `85.5/5.4`，手臂 `42.75/2.7`。训练与部署使用同一组实际数值，见 [机器人配置](../src/assets/robots/hunter180/hunter180_constants.py) 和 [接口参数 JSON](hunter180_policy_io.json)；质量比由两个本地 URDF 自动计算，最终 PD 倍数由 `HUNTER180_PD_SCALE=3.0` 指定。

这里把重量作为提高增益的依据，并使用指定的 3 倍初值；并未求解各关节的精确等效惯量。连杆长度和关节有效惯量也影响控制响应，这些不是经过实机辨识的电机增益。电机 armature 使用 Hunter130 取整后的估计值，没有按整机质量或 PD 倍数缩放。

力矩上限保留 Hunter180 URDF 的值，例如腿部主关节 450 Nm、ankle pitch 622 Nm、ankle roll 206 Nm、waist yaw/roll/pitch 400/260/356 Nm、手臂 160 Nm。`action_scale = round(0.25 * effort_limit / Kp_180, 3)`，随新的 PD 重新计算并保留三位小数，不统一使用 0.25。

## 任务与训练

```bash
conda activate mjlab
python scripts/list_envs.py --keyword Hunter180
python scripts/play.py Hunter180-Flat --agent zero --num-envs 1
python scripts/train.py Hunter180-Flat --env.scene.num-envs=4096
python scripts/train.py Hunter180-AMP-Flat --env.scene.num-envs=4096
```

已注册 `Hunter180-Flat`、`Hunter180-Rough`、`Hunter180-AMP-Flat`、`Hunter180-AMP-Rough`。普通速度任务不依赖动作文件；AMP 任务包含动作采样重置和跌倒恢复的延迟终止机制。训练日志分别位于 `logs/rsl_rl/hunter180_velocity/` 与 `logs/rsl_rl/hunter180_amp_locomotion/`。

AMP 控制周期为 20 ms（50 Hz），actor 历史为 4 帧，按时间从旧到新排列。采用 11 个实际存在的连杆构造 AMP/critic 身体观测，末端使用 elbow link，不引用不存在的 wrist link。不能直接加载 Hunter130 或 G1 的策略权重。

## 从 G1 转换的动作数据

18 段 G1 数据（17 段走跑、1 段恢复，9992 帧、50 Hz）已转换到：

```text
src/assets/motions/hunter180/amp/WalkandRun/*.npz
src/assets/motions/hunter180/amp/Recovery/*.npz
```

重新转换：

```bash
python scripts/retarget_g1_to_hunter180.py
```

转换先按名称匹配 23 个活动关节，移除 G1 双腕 roll/pitch/yaw 六个关节。Hunter180 肘关节 URDF 的固定旋转为 `rpy="0 1.57 0"`，G1 对应固定旋转为零，两者关节轴均为 +Y，所以两侧肘角采用 `q_180 = q_G1 - 1.57`，然后裁剪到 Hunter180 关节限位。

保留帧数、帧率、根节点 XY 和朝向；逐帧调整根节点 Z，使 Hunter180 的最低碰撞几何离地高度匹配 G1，并把源动作负离地高度修正为零。所有 24 个目标连杆位姿和世界坐标系连杆原点速度在 Hunter180 上重新计算，写入 joint/body names、肘部偏移和高度调整元数据。

这是关节空间的直接转换，没有优化脚/手轨迹或动力学可行性。模型的长度和关节限位不同，肘部补偿也会增加部分动作的限位裁剪；逐段统计见 [转换报告](../src/assets/motions/hunter180/amp/retarget_report.json)。实际步态和恢复能力仍需训练评估，原 G1/Hunter130 数据不能直接作为 Hunter180 的身体观测加载。

也支持已重定向的 Hunter180 CSV：

```bash
python scripts/csv_to_npz.py --robot hunter180 \
  --input-dir /path/to/hunter180_csv/WalkandRun \
  --output-dir src/assets/motions/hunter180/amp/WalkandRun \
  --input-fps 30 --output-fps 50
```

CSV 每行是 `x,y,z,qx,qy,qz,qw` 后接本模型顺序的 23 个关节角，单位为米和弧度。CSV 转换只插值并计算正向运动学；输入须已经转换为 Hunter180 关节定义，不能再次对其肘角减 1.57。

## 部署与验证

部署接口、关节参数和训练命令范围见 [Hunter180 神经网络部署文档](hunter180_deployment.md)。当前接口 JSON 来自本配置，不对应已核验的 Hunter180 训练模型；需要先训练，再核对所导出 ONNX 的实际 shape 与 run 参数。

```bash
python scripts/convert_hunter180.py
python -m unittest discover -s tests -v
python -m src.assets.robots.hunter180.hunter180_constants
```

[scene_hunter180.xml](../src/assets/robots/hunter180/xmls/scene_hunter180.xml) 用于带地面的几何预览；裸 MJCF 没有执行器，训练/Entity 配置注入位置执行器和默认站姿。

验证范围包括 URDF 质量/惯量/关节坐标和限位、指定 3 倍的 PD、足部原网格、任务注册和转换动作的模型一致性，以及 CPU 环境初始化与有限步进。它不代替 GPU 大规模训练、长时间闭环或实机验证。
