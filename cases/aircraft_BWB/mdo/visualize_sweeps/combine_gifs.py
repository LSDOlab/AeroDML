"""Combine sweep frames from Panel, RANS-GNN, and PG-RANS GNN into a single 2x3 GIF.

Layout (2 rows x 3 columns):
  - Top row: Surface field pressures (Panel Pressure, RANS-GNN, PG-RANS GNN).
  - Bottom row:
      - Column 1: Geometry parameter sliders with formatted numeric values.
      - Column 2: RANS-GNN cross-section.
      - Column 3: PG-RANS GNN cross-section.
  - Column titles at top for each model with prominent, presentation-scale text.
  - Single shared colorbar at the bottom on a clean white background.
  - Clean buffers and framing between all rows and columns.
"""
import json
import os
from pathlib import Path

# Set cache dir to avoid matplotlib cache warnings in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------ Settings ---------------------------------
SWEEPS_DIR = Path(__file__).resolve().parent / "output" / "sweeps"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODELS = ["panel", "gnn", "pgnn"]  # Left-to-right column order
MODEL_TITLES = {
    "panel": "Panel Pressure",
    "gnn": "RANS-GNN",
    "pgnn": "PG-RANS GNN",
}
OUTPUT_NAME = "combined.gif"

# Pro Presentation Dimensions & Spacing
PANEL_WIDTH = 860
PANEL_HEIGHT = 560
COL_GAP = 36          # Professional buffer between columns
ROW_GAP = 42          # Professional buffer between top & bottom rows
MARGIN_X = 40         # Outer canvas horizontal margin
MARGIN_TOP = 24       # Outer canvas top margin
HEADER_HEIGHT = 65    # Prominent column title area
COLORBAR_HEIGHT = 125 # Single shared colorbar area
MARGIN_BOTTOM = 24    # Outer canvas bottom margin

BG_COLOR = "white"        # Seamless pure white presentation background
CARD_OUTLINE = "#E2E8F0"  # Clean card border
TRACK_COLOR = "#CBD5E1"   # Clean constant slider track color
KNOB_COLOR = "#0F172A"    # High-contrast dark charcoal knob
TEXT_COLOR = "#0F172A"    # High-contrast dark charcoal text

# Presentation-scale font sizes
FONT_SIZE_TITLE = 38
FONT_SIZE_LABEL = 26
FONT_SIZE_NUM = 26

PARAMETER_LABELS = {
    "pitch": "Pitch",
    "wing_sweep": "Wing Sweep",
    "wing_position": "Wing Position",
    "wing_half_span": "Wing Half-Span",
    "centerbody_half_span": "Centerbody Half-Span",
    "transition_half_span": "Transition Half-Span",
    "wing_chord_stretch_coefficients": "Wing Chord Stretch",
    "wing_twist_coefficients": "Wing Twist",
    "airfoil_thickness_coefficients": "Airfoil Thickness",
    "airfoil_camber_coefficients": "Airfoil Camber",
    "centerbody_chord_stretch_ratio": "Centerbody Chord Stretch",
    "elevator_deflection": "Elevator Deflection",
}

PARAMETER_UNITS = {
    "pitch": "°",
    "wing_sweep": "°",
    "centerbody_half_span": " m",
    "transition_half_span": " m",
    "wing_half_span": " m",
    "wing_position": " m",
    "elevator_deflection": "°",
}
# -------------------------------------------------------------------------

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def get_font(size, bold=True):
    """Load a modern sans-serif font or fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_colorbar_image(pressure_limits, target_height):
    """Render a single shared colorbar on white background with presentation-scale text."""
    fig = plt.figure(figsize=(11, 0.95), dpi=140)
    ax = fig.add_axes([0.1, 0.45, 0.8, 0.32])
    norm = mcolors.Normalize(vmin=pressure_limits[0], vmax=pressure_limits[1])
    cb = colorbar.ColorbarBase(ax, cmap="jet", norm=norm, orientation="horizontal")
    cb.set_label("Pressure [Pa]", fontsize=16, fontweight="bold", labelpad=8)
    cb.ax.tick_params(labelsize=14)
    cb.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
    fig.patch.set_facecolor("white")

    temp_path = Path("/tmp/combined_colorbar_tmp.png")
    fig.savefig(temp_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    with Image.open(temp_path) as cb_im:
        new_w = int(cb_im.width * (target_height - 10) / cb_im.height)
        return cb_im.convert("RGB").resize((new_w, target_height - 10), RESAMPLE)


def find_varying_parameters(frames):
    """Identify which parameters change across frames and return their bounds."""
    varying = []
    if not frames:
        return varying
    for key in frames[0]:
        vals = np.array([f[key] for f in frames])
        ptp_axis0 = np.ptp(vals, axis=0)
        if np.any(ptp_axis0 > 1e-6):
            min_v = vals.min(axis=0)
            max_v = vals.max(axis=0)
            varying.append((key, min_v, max_v))
    return varying


def format_param_value(key, val):
    """Format numeric values cleanly with fixed decimals and appropriate units."""
    unit = PARAMETER_UNITS.get(key, "")
    val_arr = np.asarray(val, dtype=float)
    if val_arr.size == 1:
        num_str = f"{val_arr.item():.2f}"
    else:
        num_str = f"[{val_arr.min():.2f}, {val_arr.max():.2f}]"
    return f"{num_str}{unit}"


def render_sliders_panel(frame_data, varying, width, height, font_label, font_num):
    """Draw professional constant-color sliders with anchored numbers and prominent text."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    if not varying:
        bbox = draw.textbbox((0, 0), "Geometry Fixed", font=font_label)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, (height - (bbox[3] - bbox[1])) // 2),
                  "Geometry Fixed", fill="#4B5563", font=font_label)
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=8, outline=CARD_OUTLINE, width=1)
        return image

    num_vars = len(varying)
    padding_y = 35
    step_y = (height - 2 * padding_y) / max(num_vars, 1)

    for i, (key, min_v, max_v) in enumerate(varying):
        y_center = int(padding_y + (i + 0.5) * step_y)
        label = PARAMETER_LABELS.get(key, key.replace("_", " ").title())

        val = np.asarray(frame_data.get(key, min_v), dtype=float)
        diff = max_v - min_v
        if np.allclose(diff, 0):
            frac = 0.5
        else:
            frac = float(np.mean((val - min_v) / diff))
        frac = max(0.0, min(1.0, frac))

        num_str = format_param_value(key, val)

        x_left, x_right = 60, width - 60
        track_w = x_right - x_left
        track_y = y_center + 8

        # Parameter label on left
        draw.text((x_left, y_center - 32), label, fill=TEXT_COLOR, font=font_label)

        # Formatted number anchored right-aligned at track right
        num_bbox = draw.textbbox((0, 0), num_str, font=font_num)
        num_w = num_bbox[2] - num_bbox[0]
        draw.text((x_right - num_w, y_center - 32), num_str, fill=TEXT_COLOR, font=font_num)

        # Constant-color track
        draw.rounded_rectangle([x_left, track_y, x_right, track_y + 10], radius=5, fill=TRACK_COLOR)

        # Thumb indicator knob (sliding along the constant track)
        thumb_x = int(x_left + frac * track_w)
        r = 13
        draw.ellipse([thumb_x - r, track_y + 5 - r, thumb_x + r, track_y + 5 + r],
                     fill=KNOB_COLOR, outline="white", width=2)

    # Clean card outline
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=8, outline=CARD_OUTLINE, width=1)
    return image


def load_plot_image(folder, image_info, width, height, is_cross_section=False):
    """Load, crop out 'NN' label and colorbars, and resize into a clean card."""
    path = image_info.get("path")
    if path is None:
        blank = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(blank)
        draw.text((30, 40), "Section outside the aircraft", fill="#475569")
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=8, outline=CARD_OUTLINE, width=1)
        return blank

    with Image.open(folder / path) as original:
        img = original.convert("RGB")

        if is_cross_section:
            # Crop top (removes 'NN' label at [5:37, 8:66]) and bottom (removes scalar bar at y >= 800)
            cropped = img.crop((0, 45, original.width, 800)).resize((width, height), RESAMPLE)
            card = Image.new("RGB", (width, height), "white")
            card.paste(cropped, (0, 0))
            ImageDraw.Draw(card).rounded_rectangle([0, 0, width - 1, height - 1], radius=8, outline="#CBD5E1", width=1)
            return card
        else:
            # Surface view: erase colorbar from bottom center (x: 270..930, y: 800..895)
            # This preserves the full swept wingtip at x < 270 (which extends down to y=849).
            draw_orig = ImageDraw.Draw(img)
            draw_orig.rectangle([270, 800, 930, 895], fill="white")

            # Crop top y=45 to remove 'NN', and crop bottom at y=885 to preserve the full wingtip
            cropped = img.crop((0, 45, original.width, 885)).resize((width, height), RESAMPLE)
            card = Image.new("RGB", (width, height), "white")
            card.paste(cropped, (0, 0))
            return card


def create_combined_frame(folder, frame_idx, frames_data, varying, colorbar_image,
                          font_title, font_label, font_num):
    """Assemble a single frame containing 6 panels in a 2x3 layout with clean buffers."""
    canvas_w = 2 * MARGIN_X + 3 * PANEL_WIDTH + 2 * COL_GAP
    canvas_h = MARGIN_TOP + HEADER_HEIGHT + 2 * PANEL_HEIGHT + ROW_GAP + COLORBAR_HEIGHT + MARGIN_BOTTOM
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    top_y = MARGIN_TOP + HEADER_HEIGHT
    bot_y = top_y + PANEL_HEIGHT + ROW_GAP

    # 1. Column Titles (presentation-scale text)
    for col_idx, model in enumerate(MODELS):
        title = MODEL_TITLES.get(model, model.upper())
        col_x = MARGIN_X + col_idx * (PANEL_WIDTH + COL_GAP)
        col_center = col_x + PANEL_WIDTH // 2
        bbox = draw.textbbox((0, 0), title, font=font_title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y_title = MARGIN_TOP + (HEADER_HEIGHT - th) // 2
        draw.text((col_center - tw // 2, y_title), title, fill=TEXT_COLOR, font=font_title)

    # 2. Top Row: Surface Field Pressures for all 3 models
    for col_idx, model in enumerate(MODELS):
        col_x = MARGIN_X + col_idx * (PANEL_WIDTH + COL_GAP)
        meta = json.loads((folder / f"{model}_{frame_idx:04d}.json").read_text())
        surf_info = meta["plot_images"][0]
        surf_img = load_plot_image(folder, surf_info, PANEL_WIDTH, PANEL_HEIGHT, is_cross_section=False)
        canvas.paste(surf_img, (col_x, top_y))

    # 3. Bottom Row:
    # Column 0: Sliders showing each changing geometry parameter
    slider_panel = render_sliders_panel(frames_data[frame_idx], varying, PANEL_WIDTH, PANEL_HEIGHT, font_label, font_num)
    canvas.paste(slider_panel, (MARGIN_X, bot_y))

    # Column 1 & 2: Cross-sections for RANS-GNN and PG-RANS GNN
    for col_idx, model in enumerate(["gnn", "pgnn"], start=1):
        col_x = MARGIN_X + col_idx * (PANEL_WIDTH + COL_GAP)
        meta = json.loads((folder / f"{model}_{frame_idx:04d}.json").read_text())
        xc_info = meta["plot_images"][1]
        xc_img = load_plot_image(folder, xc_info, PANEL_WIDTH, PANEL_HEIGHT, is_cross_section=True)
        canvas.paste(xc_img, (col_x, bot_y))

    # 4. Bottom Colorbar (centered horizontally)
    cb_x = (canvas_w - colorbar_image.width) // 2
    canvas.paste(colorbar_image, (cb_x, bot_y + PANEL_HEIGHT + 10))

    return canvas


def combine_sweep_folder(folder, output_paths):
    """Combine frames of a single sweep into one 2x3 GIF."""
    manifest_path = folder / "panel.json"
    if not manifest_path.exists():
        manifest_path = folder / "gnn.json"
    manifest = json.loads(manifest_path.read_text())
    frames_data = manifest["frames"]
    num_frames = len(frames_data)
    duration = manifest.get("frame_duration_ms", 150)
    pressure_limits = manifest.get("pressure_limits", (10000.0, 40000.0))

    varying = find_varying_parameters(frames_data)
    colorbar_image = build_colorbar_image(pressure_limits, COLORBAR_HEIGHT)
    font_title = get_font(FONT_SIZE_TITLE, bold=True)
    font_label = get_font(FONT_SIZE_LABEL, bold=True)
    font_num = get_font(FONT_SIZE_NUM, bold=True)

    print(f"Combining {folder.name}: {num_frames} frames (varying: {[k for k, _, _ in varying]})...")
    combined_frames = []
    for frame_idx in range(num_frames):
        frame = create_combined_frame(folder, frame_idx, frames_data, varying,
                                      colorbar_image, font_title, font_label, font_num)
        combined_frames.append(frame)

    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_frames[0].save(
            out_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
        )
        print(f"Saved {out_path}")


def main():
    folders = sorted(path for path in SWEEPS_DIR.iterdir() if path.is_dir())
    if not folders:
        raise FileNotFoundError(f"No sweep folders in {SWEEPS_DIR}; run generate_sweep_gifs.py first")

    for folder in folders:
        for model in MODELS:
            manifest_file = folder / f"{model}.json"
            if not manifest_file.exists():
                raise FileNotFoundError(f"Missing {manifest_file}; run sweeps for {model} first")

        sweep_out = folder / OUTPUT_NAME
        root_out = OUTPUT_DIR / OUTPUT_NAME
        combine_sweep_folder(folder, [sweep_out, root_out])


if __name__ == "__main__":
    main()
