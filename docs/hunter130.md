# Hunter130

将 `/home/milkyway/opensource/EC-R130-V2_description` 中的机器人接入 AMP_mjlab，模型名称为 `hunter130`。机器人资源位于 `src/assets/robots/hunter130/`，以下命令均在仓库根目录执行。

ONNX 接口、观测历史、关节映射及实机控制对接见 [Hunter130 神经网络部署文档](hunter130_deployment.md)。

- 25 个关节：双腿 12、腰部 3、双臂 10；质量 30.87757522 kg。
- [hunter130.urdf](../src/assets/robots/hunter130/hunter130.urdf) 与同目录的 `meshes/` 保存原始机器人描述，运行不依赖原目录。
- [hunter130.xml](../src/assets/robots/hunter130/xmls/hunter130.xml) 保留 URDF 的连杆坐标、质量、完整惯量、关节轴、角度限位和力矩限位，增加浮动基座、IMU、足底 site。
- 足部视觉和碰撞均直接使用原始 `ankle_roll_link.STL`。MuJoCo 默认对 mesh 碰撞使用凸包；没有另外替换成盒体或胶囊，也没有减面。
- 其他部位使用从 STL 包围范围拟合的盒体/胶囊碰撞体，排除沿关节链相距一至两个关节的连杆碰撞，保留其他自碰撞。
- 5 个 STL 超过 MuJoCo 的 200000 三角面读取上限，生成无损索引 OBJ 用于视觉。原 STL 保留，顶点和三角形不减面。
- [机器人配置](../src/assets/robots/hunter130/hunter130_constants.py) 的 PD 直接读取项目 [G1 配置](../src/assets/robots/unitree_g1/g1_constants.py) 中对应关节的 stiffness/damping；armature 同样使用 G1 的仿真估计值。力矩限制仍来自 Hunter130 URDF，action scale 为 `0.25 * effort_limit / stiffness`。URDF 速度限制保留为配置元数据，不作为硬速度约束。

在仓库根目录执行：

```bash
conda activate mjlab
python scripts/list_envs.py --keyword Hunter130
python scripts/play.py Hunter130-Flat --agent zero --num-envs 1
python scripts/train.py Hunter130-Flat --env.scene.num-envs=4096
```

注册任务：`Hunter130-Flat`、`Hunter130-Rough`、`Hunter130-AMP-Flat`、`Hunter130-AMP-Rough`。普通速度任务不需要动作数据。

AMP 数据目录：

```text
src/assets/motions/hunter130/amp/WalkandRun/*.npz
src/assets/motions/hunter130/amp/Recovery/*.npz
```

仓库内 G1 的 18 段动作（17 段走跑、1 段恢复，共 9992 帧、50 Hz）已转换到上述目录，可以直接启动 Hunter130 AMP 训练。重新从 G1 转换：

```bash
python scripts/retarget_g1_to_hunter130.py
python scripts/train.py Hunter130-AMP-Flat --env.scene.num-envs=4096
```

转换按名称映射 25 个同名关节，移除双腕 pitch/yaw 四个关节，裁剪到 Hunter130 的关节限位。保留帧率、帧数、根节点 XY 和朝向；每帧比较两种机器人的最低碰撞几何高度，调整 Hunter130 根节点 Z 来匹配源动作的离地高度，并消除负离地高度。足部仍使用原始网格。之后在 Hunter130 模型上重新计算全部 26 个连杆的位姿和连杆原点速度，包含名称、来源及高度调整元数据。

这是按关节空间进行的直接转换，没有用 IK 锁定脚/手的轨迹，也未求解动力学可行性；两种机器人比例不同，脚滑和恢复动作的效果仍需训练验证。[转换报告](../src/assets/motions/hunter130/amp/retarget_report.json) 记录逐段限位裁剪和根节点高度调整统计。原 G1 数据保持原样，不能跳过转换直接给 Hunter130 加载。

转换实现见 [retarget_g1_to_hunter130.py](../scripts/retarget_g1_to_hunter130.py)。全量 9992 帧已通过关节限位、四元数、帧数/帧率和接地高度检查；10 项回归测试、CPU AMP 环境重置/步进及判别器数据加载检查均通过。尚未验证 GPU 训练效果。

也可以转换已经针对 Hunter130 重定向的 CSV。每帧格式为 `x,y,z,qx,qy,qz,qw` 后接 25 个关节角，单位为米和弧度，关节顺序与 `hunter130.urdf` / `HUNTER130_JOINT_NAMES` 一致：左腿、右腿、腰 yaw/roll/pitch、左臂、右臂；各腿为 hip pitch/roll/yaw、knee、ankle pitch/roll，各臂为 shoulder pitch/roll/yaw、elbow、wrist roll。

```bash
python scripts/csv_to_npz.py --robot hunter130 \
  --input-dir /path/to/hunter130_csv/WalkandRun \
  --output-dir src/assets/motions/hunter130/amp/WalkandRun \
  --input-fps 30 --output-fps 50
# Recovery 用同样的命令，输入和输出目录都改为 Recovery。
python scripts/train.py Hunter130-AMP-Flat --env.scene.num-envs=4096
```

CSV 转换脚本只做插值和该机器人上的正向运动学，不自动完成动作重定向。两种转换脚本生成的 NPZ 都包含关节和连杆名称元数据；AMP 启动时检查顺序和数组维度，缺失动作会给出明确提示。

重新生成模型与验证：

```bash
python scripts/convert_hunter130.py --source-dir /home/milkyway/opensource/EC-R130-V2_description
python -m unittest discover -s tests -v
```

`python -m src.assets.robots.hunter130.hunter130_constants` 可查看带执行器及默认站姿的机器人。[scene_hunter130.xml](../src/assets/robots/hunter130/xmls/scene_hunter130.xml) 是带地面的几何预览场景；其裸 MJCF 不包含执行器，训练时由 `EntityCfg` 注入。
