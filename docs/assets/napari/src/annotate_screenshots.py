"""
Crop and annotate the raw captures from ``capture_screenshots.py``.

Reads ``<raw>/*.png`` plus ``<raw>/geometry.json`` (widget positions recorded
at capture time, so callouts land on the real controls) and writes the figures
used by ``docs/napari_plugin.md``::

    python docs/assets/napari/src/annotate_screenshots.py RAW_DIR docs/assets/napari
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BADGE = (246, 195, 68)       # amber callout, readable on napari's dark theme
BADGE_TEXT = (20, 20, 20)
FRAME_BG = (250, 250, 250)   # light frame around composites, to sit on the site
TITLE_TEXT = (40, 40, 40)
SM_COLOUR = (255, 90, 90)
ST_COLOUR = (255, 235, 59)
MAX_WIDTH = 1600


def font(size, bold=True):
    candidates = [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def badge(draw, cx, cy, label, r=12):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BADGE, outline=(0, 0, 0), width=1)
    f = font(13 if len(str(label)) < 2 else 11)
    draw.text((cx, cy), str(label), fill=BADGE_TEXT, font=f, anchor="mm")


def trim_bottom(img, pad=12):
    """Drop the empty dock area below the last control."""
    arr = np.asarray(img.convert("RGB"), dtype=int)
    bg = arr[-5, 5]
    diff = np.abs(arr - bg).sum(axis=2) > 30
    rows = np.where(diff.any(axis=1))[0]
    last = int(rows.max()) if rows.size else arr.shape[0] - 1
    return img.crop((0, 0, img.width, min(img.height, last + pad)))


def annotate_panel(img, boxes, labels, margin=40):
    """Add a left margin with numbered badges aligned to each control row."""
    img = trim_bottom(img)
    bg = img.convert("RGB").getpixel((5, img.height - 3))
    out = Image.new("RGB", (img.width + margin, img.height), bg)
    out.paste(img, (margin, 0))
    draw = ImageDraw.Draw(out)
    for key, label in labels.items():
        x, y, w, h = boxes[key]
        cy = y + h / 2
        badge(draw, margin / 2, cy, label)
        draw.line([(margin / 2 + 12, cy), (margin + 4, cy)], fill=BADGE, width=2)
    return out


def titled(img, title, pad=10, height=34):
    f = font(16)
    out = Image.new("RGB", (img.width + 2 * pad, img.height + height + pad), FRAME_BG)
    draw = ImageDraw.Draw(out)
    draw.text((pad, height / 2 + 2), title, fill=TITLE_TEXT, font=f, anchor="lm")
    out.paste(img, (pad, height))
    return out


def side_by_side(images, gap=12):
    h = max(i.height for i in images)
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), FRAME_BG)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def save(img, path):
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    img = img.convert("RGB")
    img.save(path, optimize=True)
    if path.stat().st_size > 900_000:
        img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"{path.name}: {img.width}x{img.height}, {path.stat().st_size / 1e3:.0f} kB")


def crop_box(img, box):
    x, y, w, h = box
    return img.crop((x, y, x + w, y + h))


def main(raw, out):
    raw, out = Path(raw), Path(out)
    geo = json.loads((raw / "geometry.json").read_text())
    load = lambda name: Image.open(raw / name).convert("RGB")  # noqa: E731

    # Overview: four docks, outlined and numbered.
    img = load("00_overview_raw.png")
    draw = ImageDraw.Draw(img)
    for i, (name, (x, y, w, h)) in enumerate(geo["00_overview_raw"].items(), start=1):
        draw.rounded_rectangle([x + 2, y + 2, x + w - 3, y + h - 3], radius=6, outline=BADGE, width=3)
        badge(draw, x - 18, y + 20, i, r=14)
    save(img, out / "overview.png")

    # 1. Load
    save(annotate_panel(load("01_load_panel.png"), geo["01_load_panel"], {
        "source_file": 1, "target_file": 2, "pixel_size": 3,
        "show_density": 4, "call_button": 5,
    }), out / "load_panel.png")

    # 2. Pre-register: before / after canvases, then the annotated panel.
    before = crop_box(load("02_prereg_before_window.png"), geo["02_prereg_before_window"]["canvas"])
    after = crop_box(load("02_prereg_window.png"), geo["02_prereg_window"]["canvas"])
    scale = 640 / before.width
    before = before.resize((640, round(before.height * scale)), Image.LANCZOS)
    after = after.resize((640, round(after.height * scale)), Image.LANCZOS)
    save(side_by_side([
        titled(before, "Update preview (rotation 0): overlap 0.596"),
        titled(after, "Search rotation/flip (rotation 90): overlap 0.953"),
    ]), out / "prereg_before_after.png")
    save(annotate_panel(load("02_prereg_panel.png"), geo["02_prereg_panel"], {
        "source": 1, "target": 2, "rotation": 3, "flip_x": 4, "flip_y": 5,
        "scale_mode": 6, "manual_scale": 7, "preserve_aspect": 8,
        "preview": 9, "search": 10, "apply": 11, "status": 12,
    }), out / "prereg_panel.png")

    # 3. Landmarks: number each pair on the canvas in click order.
    canvas = load("03_landmarks_canvas.png")
    xy = np.load(raw / "03_landmarks_canvas_xy.npy", allow_pickle=True).item()
    draw = ImageDraw.Draw(canvas)
    f = font(17)
    for key, colour, dx in (("sm", SM_COLOUR, -14), ("st", ST_COLOUR, 14)):
        anchor = "rm" if dx < 0 else "lm"
        for i, (x, y) in enumerate(xy[key], start=1):
            draw.text((x + dx, y), str(i), fill=colour, font=f, anchor=anchor,
                      stroke_width=3, stroke_fill=(0, 0, 0))
    lf = font(15)
    draw.rounded_rectangle([10, 10, 262, 66], radius=6, fill=(30, 30, 30), outline=(90, 90, 90))
    draw.ellipse([20, 20, 34, 34], fill=(255, 0, 0), outline=(255, 255, 255))
    draw.text((42, 27), "SM landmarks (source)", fill=(235, 235, 235), font=lf, anchor="lm")
    draw.ellipse([20, 42, 34, 56], fill=(255, 255, 0), outline=(255, 255, 255))
    draw.text((42, 49), "ST landmarks (target)", fill=(235, 235, 235), font=lf, anchor="lm")
    save(canvas, out / "landmarks_canvas.png")
    save(annotate_panel(load("03_landmarks_panel.png"), geo["03_landmarks_panel"], {
        "create": 1, "check": 2, "save": 3, "load": 4, "status": 5,
    }), out / "landmarks_panel.png")

    # 4. Register: st_sm and centroid panels side by side.
    st_sm_labels = {
        "round": 1, "backend": 2, "source_file": 3, "source_x": 4,
        "target_file": 5, "target_x": 6, "source_points": 7, "target_points": 8,
        "output_dir": 9, "niter": 10, "partition": 11, "cpus": 12, "memory": 13,
        "time_limit": 14, "gpus": 15, "gpu_type": 16, "submit": 17, "status": 18,
    }
    st_sm = annotate_panel(load("04_register_panel_st_sm.png"), geo["04_register_panel_st_sm"], st_sm_labels)
    centroid = annotate_panel(load("04_register_panel_centroid.png"), geo["04_register_panel_centroid"], {
        "round": 1, "method": 19, "max_distance": 20,
    })
    save(side_by_side([
        titled(st_sm, "Round st_sm: sequential sections, sbatch on one GPU"),
        titled(centroid, "Round centroid: same section, CPU"),
    ]), out / "register_panels.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
