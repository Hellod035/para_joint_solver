"""Screen-space labels with leaders for the inspector's 3D annotations."""

import numpy as np


def project(position, camera, width, height):
    """Project a base-frame point through MuJoCo's mono (averaged stereo) camera."""
    relative = np.asarray(position) - camera.pos
    depth = np.dot(relative, camera.forward)
    if depth <= camera.frustum_near:
        return None
    x = np.dot(relative, np.cross(camera.forward, camera.up))
    y = np.dot(relative, camera.up)
    if not camera.orthographic:
        x *= camera.frustum_near / depth
        y *= camera.frustum_near / depth
    vertical = camera.frustum_top - camera.frustum_bottom
    horizontal = vertical * width / height
    px = width * ((x - camera.frustum_center) / horizontal + 0.5)
    py = height * (camera.frustum_top - y) / vertical
    if 0 <= px < width and 0 <= py < height:
        return np.array([px, py])
    return None


def draw_labels(image, labels, camera):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=13)
    width, height = image.size
    placed = []
    for position, text, rgba, leader in reversed(labels):
        anchor = project(position, camera, width, height)
        if anchor is None:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        box_width, box_height = right - left + 10, bottom - top + 8
        candidates = []
        for distance in ((12, 32, 60, 95, 135, 180) if leader else (6, 20, 36, 60)):
            for dx, dy in ((distance, -box_height / 2), (-box_width - distance, -box_height / 2),
                           (-box_width / 2, -box_height - distance), (-box_width / 2, distance),
                           (distance, -box_height - distance), (-box_width - distance, distance)):
                x = float(np.clip(anchor[0] + dx, 4, width - box_width - 4))
                y = float(np.clip(anchor[1] + dy, 4, height - box_height - 4))
                box = (x, y, x + box_width, y + box_height)
                overlap = sum(
                    max(0, min(box[2] + 3, b[2]) - max(box[0] - 3, b[0]))
                    * max(0, min(box[3] + 3, b[3]) - max(box[1] - 3, b[1])) for b in placed
                )
                center = np.array([x + box_width / 2, y + box_height / 2])
                candidates.append((overlap * 10000 + np.linalg.norm(center - anchor), box))
        _, box = min(candidates, key=lambda candidate: candidate[0])
        placed.append(box)
        rgb = tuple(int(255 * np.clip(c, 0, 1)) for c in rgba[:3])
        if leader:
            end = (np.clip(anchor[0], box[0], box[2]), np.clip(anchor[1], box[1], box[3]))
            draw.line((tuple(anchor), end), fill=rgb + (210,), width=1)
        draw.ellipse((anchor[0] - 2, anchor[1] - 2, anchor[0] + 2, anchor[1] + 2),
                     fill=rgb + (255,))
        draw.rounded_rectangle(box, radius=3, fill=(12, 24, 36, 230), outline=rgb + (160,))
        draw.text((box[0] + 5 - left, box[1] + 4 - top), text, fill=(239, 246, 252, 255),
                  font=font)


def draw_frame_axes(image, axes, camera):
    """Keep selected coordinate frames visible through meshes with thin arrows."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    for origin, end, rgba in axes:
        a, b = (project(p, camera, *image.size) for p in (origin, end))
        if a is None or b is None:
            continue
        direction = b - a
        norm = np.linalg.norm(direction)
        if norm < 1:
            continue
        direction /= norm
        side = np.array([-direction[1], direction[0]])
        color = tuple(int(c * 255) for c in rgba)
        draw.line((tuple(a), tuple(b)), fill=color, width=2)
        draw.polygon([tuple(b), tuple(b - 5 * direction + 2 * side),
                      tuple(b - 5 * direction - 2 * side)], fill=color)
        draw.ellipse((a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3),
                     fill=(239, 246, 252, 255), outline=(16, 24, 35, 255))
