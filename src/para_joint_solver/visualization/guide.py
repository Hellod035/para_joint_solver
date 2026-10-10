"""Parameter explanations from the loaded mechanism, in public coordinates."""

import numpy as np
import yaml


def mechanism_guide(mechanism):
    groups = []

    def group(title):
        entries = []
        groups.append(dict(title=title, entries=entries))
        return entries

    def entry(entries, path, title, values, components, unit, meaning, target):
        entries.append(dict(
            path=path, title=title, values=np.atleast_1d(values).tolist(),
            components=components, unit=unit, meaning=meaning, target=target,
        ))

    def joint_entries(entries, joint, prefix, target):
        entry(entries, prefix + ".position", "转轴上的参考点", joint.position,
              ["X", "Y", "Z"], "m",
              "参考姿态下，转轴上一点在机构基坐标系中的绝对坐标。"
              "它与 axis 共同确定一条转轴；不是相对上一关节的位移。", target)
        entry(entries, prefix + ".axis", "转轴正方向", joint.axis,
              ["X 分量", "Y 分量", "Z 分量"], "",
              "参考姿态下、用基坐标系表示的单位方向向量。长度必须为 1；"
              "正物理转角遵循右手定则。负分量表示朝该坐标轴的负方向。", target)
        entry(entries, prefix + ".scale", "公共读数到物理转角的倍率", joint.scale,
              ["倍率"], "",
              f"物理转角 = scale × (公共读数 − 参考读数)。此关节参考读数为 "
              f"{joint.reference:g} rad。负值反向；绝对值改变倍率；省略时为 1。", target)
        if np.isfinite(joint.limits).all():
            entry(entries, prefix + ".limits", "公共角度读数限位", joint.limits,
                  ["下限", "上限"], "rad",
                  "按此处的公共坐标读数限制运动，并非相对参考姿态的增量。"
                  "同时还须满足两条连杆闭合、装配分支和其他关节限位。", target)
        else:
            entries.append(dict(
                path=prefix + ".limits", title="未设置独立角度限位", values=[],
                components=[], unit="", target=target,
                meaning="此字段未配置。可达范围仍受连杆几何、电机限位和奇异位形约束；"
                        "滑块显示范围不代表完整可达范围。",
            ))

    for i, joint in enumerate(mechanism.output):
        entries = group(f"输出关节 {i + 1} · {joint.name}")
        entry(entries, f"output.joints[{i}].name", "公共坐标名称", joint.name,
              ["名称"], "", "roll 为横滚、pitch 为俯仰；名称确定公共 RP 数组中的坐标。"
              "实际旋转方向由 axis 决定，列表顺序决定从基座到平台的物理关节顺序。",
              f"joint:{joint.name}")
        joint_entries(entries, joint, f"output.joints[{i}]", f"joint:{joint.name}")
    for branch in mechanism.branches:
        name = branch.motor.name
        prefix = f"branches.{name}"
        entries = group(f"支链 {name} · 电机、曲柄与连杆")
        joint_entries(entries, branch.motor, prefix + ".motor", f"motor:{name}")
        entry(entries, prefix + ".crank_tip", f"曲柄端连接点 B_{name}", branch.crank_tip,
              ["X", "Y", "Z"], "m",
              "参考姿态下，曲柄与连杆连接中心的基坐标。随电机曲柄运动；"
              "与电机轴线的垂直距离决定曲柄半径。这里 B 表示连接点，不是电机 B。",
              f"tip:{name}")
        entry(entries, prefix + ".platform_anchor", f"平台端连接点 C_{name}",
              branch.platform_anchor, ["X", "Y", "Z"], "m",
              "参考姿态下，连杆与输出平台连接中心的基坐标。该点固定在输出平台上，"
              "运动时依次受到两个输出关节的变换。", f"anchor:{name}")
        entry(entries, prefix + ".rod_length", f"连杆 B_{name} → C_{name} 的中心距",
              branch.rod_length, ["长度"], "m",
              "两连接中心之间的固定距离，运动中保持不变。参考姿态应满足 "
              "‖platform_anchor − crank_tip‖ = rod_length。不是 STL 外形包围盒的长度。",
              f"rod:{name}")
    entries = group("装配参考读数 · assembly")
    entry(entries, "assembly.home_rp", "输出关节参考读数", mechanism.home_rp,
          ["Roll", "Pitch"], "rad",
          "所有轴线与连接点坐标对应此输出读数。顺序始终是 [roll, pitch]，"
          "与 output.joints 的物理排列顺序无关。也是默认正解初值和 Reset Home 姿态。",
          "output")
    entry(entries, "assembly.motor_reference", "电机参考读数", mechanism.motor_reference,
          ["A", "B"], "rad",
          "同一参考姿态下的电机读数，顺序固定为 [A, B]。它们与参考曲柄端点一起"
          "确定曲柄装配分支；读数可以非零。", "motors")
    source = yaml.safe_load(mechanism.source_yaml) if mechanism.source_yaml else {
        "id": mechanism.id, "topology": "dual_crank_two_rod",
    }
    meanings = dict(
        id="机构标识，用于命令行选择、查看器标题与报告。",
        topology="机构拓扑：双曲柄、双连杆、两个旋转输出关节。",
        description="机构说明与简化假设；不参与几何计算。",
        sources="几何参考资料来源；不参与求解。",
    )
    metadata = [dict(path=key, value=source[key], meaning=meaning)
                for key, meaning in meanings.items() if key in source]
    return dict(
        id=mechanism.id, topology="dual_crank_two_rod",
        output_order=[j.name for j in mechanism.output], groups=groups, metadata=metadata,
        source_yaml=mechanism.source_yaml,
    )
