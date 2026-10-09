import json
import shutil

import pytest

from para_joint_solver.cli import main
from para_joint_solver import load_solver


def test_solve_cli(capsys):
    assert (
        main(
            [
                "solve",
                "g1",
                "--q-ab",
                "0",
                "0",
                "--dq-ab",
                ".1",
                "-.1",
                "--tau-rp",
                "4",
                "2",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert len(data["dq_rp"]) == len(data["tau_ab"]) == 2


def test_error_exit_status(capsys):
    assert main(["solve", "g1", "--q-ab", "20", "0"]) == 2
    assert "outside" in capsys.readouterr().err


def test_core_only_report(tmp_path):
    assert (
        main(
            [
                "validate",
                "--robot",
                "g1",
                "--core-only",
                "--grid",
                "3",
                "--random",
                "0",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "summary.json").read_text())["passed"]
    assert (tmp_path / "index.html").exists()


def test_benchmark_plots_numerical_errors_without_meshes_or_renderer(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("matplotlib")

    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark must not create a MuJoCo renderer")

    monkeypatch.setattr(mujoco, "Renderer", forbidden)
    robot = tmp_path / "g1"
    robot.mkdir()
    shutil.copyfile(load_solver("g1").mechanism.directory / "mechanism.yaml", robot / "mechanism.yaml")
    output = tmp_path / "report"
    assert main([
        "benchmark", "--robot", str(robot), "--plot", "--grid", "3", "--random", "0",
        "--samples", "2", "--repeats", "1", "--output", str(output),
    ]) == 0
    data = json.loads((output / "g1" / "report.json").read_text())
    assert data["passed"] and data["plots"] == ["errors.png"]
    assert "mujoco.fk_rad" in data["checks"]
    assert not any("mujoco" in name for name in data["benchmark"]["timings"])
    assert len(data["samples"]) == data["sample_count"]
    for name, check in data["checks"].items():
        values = [sample["values"][name] for sample in data["samples"] if name in sample["values"]]
        assert max(values) == check["max_error"]
    assert (output / "g1" / "errors.png").stat().st_size > 1000
    html = (output / "index.html").read_text()
    assert "Per-pose numerical errors" in html
    assert "Core solver latency" in html and "mujoco_fk" not in html
