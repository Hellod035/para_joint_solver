"""One consistent diagnostic appearance for every robot; not robot configuration."""

STYLE = dict(marker_radius=0.0035)


def frame_color(frame):
    if frame.startswith("motor_"):
        return (1.0, 0.6, 0.15, 1.0)
    if frame.startswith("rod_"):
        return (0.72, 0.77, 0.86, 1.0)
    if frame == "base":
        return (0.45, 0.53, 0.62, 1.0)
    return (0.16, 0.58, 0.78, 1.0)
