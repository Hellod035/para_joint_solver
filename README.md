# para-joint-solver

可配置的双曲柄、双连杆并联关节求解器，支持使用 MuJoCo 交叉验证。
提供 AB 角度到 RP 角度、AB 角速度到 RP 角速度，以及 RP 力矩到 AB 力矩的映射。
所有物理量均使用国际单位制。

```bash
uv sync --all-extras --locked
uv run pjs list
```

现有示例包括 `g1`、`tiangong`、`igris_c_waist`、`limx_waist`、`waist_offset` 和 `waist_symmetric`。
对称腰部采用相交的 roll → pitch 输出轴和等长连杆；
偏置腰部采用错位的 pitch → roll 输出轴和非对称连杆。

## 网页可视化

```bash
uv run pjs inspect g1
```

本地网页提供实时 AB/RP 滑块、关节限位与剩余行程、网格显示，以及与 MuJoCo 的数值对比。
无效输入会保留上一次有效姿态。拖动视图可旋转相机，滚轮缩放，双击恢复默认视角。
服务默认从端口 8765 启动，也可用 `--port 8766` 指定起始端口；若端口被占用，
会自动尝试更高端口，并打印实际地址、在浏览器中打开。`--port 0` 由操作系统分配端口，
按 Ctrl-C 停止服务。

RP 滑块的总范围保持固定，灰色区间表示另一个角度固定时的可行范围，拖动会停在其边界。
数值输入会按输入值尝试求解，超出限制时报告错误。页面显示的解算时间仅包含核心位置、
雅可比矩阵、速度和力矩计算。

可视化展示机构的运动学姿态，主要零件包括基座平台、输出平台、两个曲柄和两根连杆。
LimX 示例采用六个原始 STL，中间转动件不显示。

使用 `uv run pjs inspect limx_waist` 可打开 LimX HU_D04_01 腰部。
该示例固定上躯干、让下腰相对运动，保留官方 MJCF 的角度读数，排除独立 yaw。
几何和 STL 坐标系已与原始 MJCF 交叉验证，未包含实机标定。
详见 [LimX 腰部坐标约定](docs/mechanism.md#limx-腰部示例)及[来源说明](robots/limx_waist/NOTICE)。

坐标系开关可以显示固定的机构基坐标轴，以及各 STL 经 `visuals.yaml` 装配后的自身坐标轴。
X、Y、Z 分别使用红、绿、蓝色。右侧 **各 STL 自身坐标系** 面板为每个文件提供独立开关，
可同时打开多个零件，也可通过 **全部显示** 统一切换。
STL 坐标系默认隐藏，显示后会随零件运动；短小、无文字的坐标轴可透过网格观察，
长度仅用于示意，不代表 STL 的单位长度。

可选的几何标记用于显示关节轴方向、曲柄端连接点（`B_A`、`B_B`）和平台连接点（`C_A`、`C_B`），
这些叠加显示不会改变机构参数或解算结果。组装基坐标轴的原点是 `mechanism.yaml` 使用的
固定基准 `(0,0,0)`，不一定处于模型中心；它与各零件自身的 STL 坐标系分别显示。

右侧 **参数可视化** 页签解释全部机构字段，包括向量分量、单位、限位、倍率、参考读数、
名称和来源信息。数值标记默认隐藏，每个参数按钮可独立显示坐标点及 XYZ 数值、轴向量、
杆长或角度限位圆弧。多个按钮可同时开启，**隐藏全部** 可清除所有参数标记。
面板还提供求解器实际加载的完整 YAML 内容。

黄色参数标记保持在参考几何位置，STL 坐标系和紫色关节轴则跟随机构运动。
点击复位按钮（**Reset Home**）可使模型回到参考姿态。限位圆弧考虑关节的有符号 `scale`，
其半径仅用于示意。场景标签保留七位有效数字，参数面板保留加载数值的完整精度。
无显示设备的 Linux 环境可使用 `MUJOCO_GL=egl uv run pjs inspect igris_c_waist`。

## 基准测试

```bash
uv run pjs benchmark --all --plot
uv run pjs benchmark --robot waist_symmetric --samples 300 --repeats 5
```

检查内容包括闭环误差、位置、速度、力矩映射、有限差分，以及 MuJoCo 闭链雅可比矩阵。
性能统计报告核心求解器在逆解（IK）、正解（FK）、雅可比矩阵求值、速度与力矩映射上的
中位数和 P95 耗时。MuJoCo 仅用于交叉验证，不计入核心解算时间。

打开 `artifacts/benchmark/index.html` 查看报告。`--plot` 会添加各姿态的位置、闭环、
速度、力矩、雅可比矩阵和虚功率误差图。基准测试不需要网格或 OpenGL 渲染。
每个机构的 JSON 文件及 `summary.json` 保存测试结果，任一检查失败都会返回非零退出码。

## 新增机构与配置说明

新增机构需要 `robots/<name>/mechanism.yaml`、`visuals.yaml` 和 `meshes/`。
[机构配置说明](docs/mechanism.md) 提供完整字段定义和可运行的示例。

常用说明入口：[全部配置字段的物理含义](docs/mechanism.md#mechanismyaml-各字段的物理含义)、
[position 与 axis](docs/mechanism.md#position-和-axis-的物理含义)、
[axis 对 position 哪个分量不敏感](docs/mechanism.md#axis-对-position-哪个分量不敏感)、
[STL 装配位置和旋转](docs/mechanism.md#motor_a-的装配位置和旋转)、
[省略 pose 和 scale 的导出要求](docs/mechanism.md#省略-pose-和-scale-的-stl-导出要求)、
[连杆绕 Z 的朝向](docs/mechanism.md#连杆绕-z-的朝向是什么意思)。

机构配置在统一的基坐标系中定义轴线和参考连接点，仅凭该文件即可解算。
网格装配和检查采样范围在 `visuals.yaml` 中配置。RP 限位可选，检查采样范围不代表完整可达边界。
`external/` 用于存放可选的上游参考资料，已被 Git 忽略。
