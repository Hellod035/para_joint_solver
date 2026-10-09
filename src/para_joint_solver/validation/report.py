"""Portable HTML report generation."""

from pathlib import Path
import html


def write_html(reports, directory):
    directory = Path(directory)
    cards = []
    for report in reports:
        robot = html.escape(report["robot"])
        rows = "".join(
            f"<tr><td>{html.escape(n)}</td><td>{v['max_error']:.3e}</td>"
            f"<td>{v['tolerance']:.3e}</td><td>{'PASS' if v['passed'] else 'FAIL'}</td></tr>"
            for n, v in report.get("checks", {}).items()
        )
        failures = "".join(f"<li>{html.escape(str(f))}</li>" for f in report.get("failures", []))
        images = ""
        benchmark = ""
        if "benchmark" in report:
            timings = report["benchmark"]["timings"]
            timing_rows = "".join(
                f"<tr><td>{html.escape(n)}</td><td>{v['median_us']:.1f}</td><td>{v['p95_us']:.1f}</td></tr>"
                for n, v in timings.items()
            )
            benchmark = f"<h3>Core solver latency / microseconds</h3><table><tr><th>Operation</th><th>Median</th><th>P95</th></tr>{timing_rows}</table>"
        images = "".join(
            f'<img src="{robot}/{html.escape(filename)}" alt="Per-pose numerical errors">'
            for filename in report.get("plots", [])
        )
        cards.append(
            f"<section><h2>{robot} · {'PASS' if report['passed'] else 'FAIL'}</h2>"
            f"<p>{report.get('sample_count', 0)} numerical poses; JSON: <a href='{robot}/report.json'>report</a></p>"
            f"{images}{benchmark}<details><summary>Numerical checks</summary><table><tr><th>Check</th><th>Max error</th><th>Tolerance</th><th>Status</th></tr>{rows}</table></details><ul>{failures}</ul></section>"
        )
    text = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Parallel joint validation</title>
<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#182737;background:#edf1f5}section{background:white;border-radius:12px;padding:24px;margin:24px 0}img{width:100%;height:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dde3ea;text-align:left}summary{cursor:pointer;font-weight:600}a{color:#086aa4}</style>
<h1>Parallel joint validation</h1><p>Public coordinates: RP=[roll,pitch], AB=[A,B]. J=dAB/dRP. SI units.</p>
<p>Error plots show absolute errors per validation pose. Dashed lines mark acceptance tolerances; red crosses mark failed samples. Torque maps use virtual power. Latency tables measure only the core solver; velocity and torque maps reuse a solved state. MuJoCo provides kinematic cross-validation without timing, dynamics stepping or mesh rendering. Use the browser inspector to review the mechanism visually.</p>
"""
        + "".join(cards)
        + "</html>"
    )
    path = directory / "index.html"
    path.write_text(text)
    return path
