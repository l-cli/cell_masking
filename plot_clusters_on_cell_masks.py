#!/usr/bin/env python
"""
Utilities to stitch HoVer-Net JSON tiles into whole-slide coordinates and
plot clustering results restricted to nuclei / cell contours.

Core entrypoints:
    plot_clusters_on_cell_masks(...)
    plot_numerical_values_on_cell_masks(...)
    plot_values_on_cell_masks(...)
    plot_selected_cluster_mask_on_he(...)

Typical use:
    from plot_clusters_on_cell_masks import (
        plot_clusters_on_cell_masks,
        plot_numerical_values_on_cell_masks,
        plot_values_on_cell_masks,
        plot_selected_cluster_mask_on_he,
    )
"""

import os
import re
import glob
import json
import gc

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

try:
    import pyvips
except ImportError as e:
    raise ImportError(
        "pyvips is required for this script. Install with `pip install pyvips` "
        "and ensure libvips is available on your system."
    ) from e

from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None  # avoid DecompressionBombError for large WSIs

# ---------------------------------------------------------------------
# Helpers: color, legend, saving
# ---------------------------------------------------------------------


def _to_rgb_u8(col):
    """'#RRGGBB' or '#RGB' or (R,G,B) -> np.uint8[3]."""
    if isinstance(col, str) and col.startswith("#"):
        h = col[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return np.array(
            [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
            dtype=np.uint8,
        )
    return np.array(col, dtype=np.uint8)


def _to_rgb01(col_u8_tuple):
    r, g, b = col_u8_tuple
    return (r / 255.0, g / 255.0, b / 255.0)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _get_font(font_px):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_px)
    except Exception:
        try:
            import matplotlib.font_manager as fm

            fpath = fm.findfont("DejaVu Sans", fallback_to_default=True)
            return ImageFont.truetype(fpath, font_px)
        except Exception:
            return ImageFont.load_default()


def _text_size_pil(draw, txt, font):
    try:
        bbox = draw.textbbox((0, 0), str(txt), font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        return draw.textsize(str(txt), font=font)


def _add_title_above_image(
    pil_img,
    title,
    background_color=(255, 255, 255),
    text_color=(0, 0, 0),
    title_font_rel=0.035,
    title_min_font_px=18,
    title_pad_rel=0.012,
):
    """
    Add a title strip above the image and return the combined image.
    """
    if title is None or str(title).strip() == "":
        return pil_img

    base = pil_img.convert("RGB")
    W, H = base.size

    font_px = max(int(round(H * float(title_font_rel))), int(title_min_font_px))
    pad = max(6, int(round(H * float(title_pad_rel))))
    font = _get_font(font_px)

    tmp = Image.new("RGB", (10, 10), tuple(int(x) for x in background_color))
    draw_tmp = ImageDraw.Draw(tmp)
    tw, th = _text_size_pil(draw_tmp, title, font)

    title_h = th + 2 * pad

    canvas = Image.new(
        "RGB",
        (W, H + title_h),
        tuple(int(x) for x in background_color),
    )
    canvas.paste(base, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    tx = max(0, (W - tw) // 2)
    ty = pad
    draw.text((tx, ty), str(title), fill=tuple(int(x) for x in text_color), font=font)

    return canvas


def _stack_images_vertical(
    top_img,
    bottom_img,
    background_color=(255, 255, 255),
    gap_px=12,
    center=True,
):
    """
    Stack two PIL images vertically without cropping.
    """
    top = top_img.convert("RGB")
    bottom = bottom_img.convert("RGB")

    W = max(top.width, bottom.width)
    H = top.height + gap_px + bottom.height

    canvas = Image.new("RGB", (W, H), tuple(int(x) for x in background_color))

    top_x = (W - top.width) // 2 if center else 0
    bottom_x = (W - bottom.width) // 2 if center else 0

    canvas.paste(top, (top_x, 0))
    canvas.paste(bottom, (bottom_x, top.height + gap_px))

    return canvas


def _add_legend_outside_right_transparent(
    pil_img,
    items,
    legend_font_rel=0.04,
    legend_min_font_px=16,
    text_color=None,
    strip_background_color=None,
):
    """
    Add a legend strip to the RIGHT of the image (top-right aligned).

    Parameters
    ----------
    pil_img : PIL.Image
    items : list[(label, (R,G,B))]
    text_color : RGBA tuple or None
    strip_background_color : None or RGB/RGBA tuple
        If None, the strip stays transparent.
        If provided, the strip background is filled with this color so it matches
        the overall plot background.
    """
    if text_color is None:
        text_color = (255, 255, 255, 255)

    if not items:
        return pil_img

    base = pil_img.convert("RGBA")
    W, H = base.size

    text_px = max(int(round(H * float(legend_font_rel))), int(legend_min_font_px))
    swatch = max(12, int(round(text_px * 0.9)))
    gap = max(6, int(round(text_px * 0.5)))
    line_gap = max(3, int(round(text_px * 0.3)))
    box_pad = max(8, int(round(text_px * 0.6)))
    strip_pad = max(8, int(round(text_px * 0.6)))

    font = _get_font(text_px)
    draw_base = ImageDraw.Draw(base)

    def _text_size(txt):
        return _text_size_pil(draw_base, txt, font)

    widths, heights = zip(*(_text_size(lbl) for lbl, _ in items)) if items else ([40], [text_px])
    text_h = max(heights) if heights else text_px
    row_h = max(swatch, text_h) + line_gap
    content_w = swatch + gap + (max(widths) if widths else 40)
    content_h = len(items) * row_h - line_gap

    legend_w = box_pad * 2 + content_w
    legend_h = box_pad * 2 + content_h

    strip_w = strip_pad * 2 + legend_w
    new_W = W + strip_w
    new_H = max(H, strip_pad * 2 + legend_h)

    if strip_background_color is None:
        canvas = Image.new("RGBA", (new_W, new_H), (0, 0, 0, 0))
    else:
        bg = tuple(int(x) for x in strip_background_color)
        if len(bg) == 3:
            bg = bg + (255,)
        canvas = Image.new("RGBA", (new_W, new_H), bg)

    canvas.alpha_composite(base, (0, 0))

    drawp = ImageDraw.Draw(canvas, "RGBA")

    x0 = W + strip_pad + box_pad
    y0 = strip_pad + box_pad

    y = y0
    for lbl, col in items:
        drawp.rectangle(
            [x0, y, x0 + swatch, y + swatch],
            fill=(int(col[0]), int(col[1]), int(col[2]), 255),
            outline=None,
        )
        tx = x0 + swatch + gap
        ty = y + max(0, (swatch - text_h) // 2)
        drawp.text((tx, ty), str(lbl), fill=text_color, font=font)
        y += row_h

    return canvas


def _save_with_matplotlib(pil_img, out_base, formats, dpi, face_rgb01):
    """Save PIL image to multiple formats using Matplotlib (enables PDF/SVG/etc.)."""
    arr = np.array(pil_img)
    H, W = arr.shape[:2]
    fig_w_in = W / float(dpi)
    fig_h_in = H / float(dpi)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor(face_rgb01)
    ax.set_facecolor(face_rgb01)
    ax.imshow(arr)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    saved_paths = []
    for fmt in formats:
        ext = str(fmt).lower().lstrip(".")
        out_path = f"{out_base}.{ext}"
        fig.savefig(
            out_path,
            dpi=dpi,
            facecolor=face_rgb01,
            edgecolor=face_rgb01,
            bbox_inches=None,
            pad_inches=0,
            format=ext,
        )
        print(f"✅ Saved: {out_path}")
        saved_paths.append(out_path)

    plt.close(fig)
    return saved_paths


def _ensure_vips_rgb_u8(vimg: "pyvips.Image") -> "pyvips.Image":
    """Ensure vips image is 3-band uint8 RGB."""
    if vimg.bands > 3:
        vimg = vimg[0:3]
    elif vimg.bands == 1:
        vimg = vimg.bandjoin([vimg, vimg, vimg])

    if vimg.format != "uchar":
        vimg = vimg.cast("uchar")
    return vimg


def _vips_to_pil_rgb(vimg: "pyvips.Image") -> Image.Image:
    """Convert a (small-ish) vips RGB uchar image to PIL.Image."""
    vimg = _ensure_vips_rgb_u8(vimg)
    mem = vimg.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(vimg.height, vimg.width, vimg.bands)
    return Image.fromarray(arr, mode="RGB")


def _scale_polygon_xy(poly_yx: np.ndarray, scale: float):
    """
    poly_yx: (M,2) in (y,x) full-res.
    returns list[(x,y)] scaled for PIL drawing in downsampled space.
    """
    if poly_yx.size == 0:
        return []
    return [(float(x) * scale, float(y) * scale) for (y, x) in poly_yx]


def _labels_equal(a, b) -> bool:
    """Robust label equality for mixed dtypes / categoricals."""
    try:
        return a == b
    except Exception:
        return str(a) == str(b)


def _pil_desaturate_and_whiten(rgba_img: Image.Image, desaturate=0.85, whiten=0.35) -> Image.Image:
    """
    Make an RGBA image look 'faded':
      - desaturate in [0,1]: 0 = keep color, 1 = fully grayscale
      - whiten in [0,1]: 0 = no whitening, 1 = fully white
    """
    if rgba_img.mode != "RGBA":
        rgba_img = rgba_img.convert("RGBA")

    rgb = rgba_img.convert("RGB")

    color_factor = max(0.0, min(1.0, 1.0 - float(desaturate)))
    rgb_desat = ImageEnhance.Color(rgb).enhance(color_factor)

    whiten = max(0.0, min(1.0, float(whiten)))
    white = Image.new("RGB", rgb_desat.size, (255, 255, 255))
    rgb_faded = Image.blend(rgb_desat, white, whiten)

    a = rgba_img.split()[-1]
    out = rgb_faded.convert("RGBA")
    out.putalpha(a)
    return out


def _auto_text_color_for_bg(bg_color, threshold=186):
    """
    Return white text for dark backgrounds, black text for light backgrounds.
    """
    rgb = _to_rgb_u8(bg_color).astype(float)
    r, g, b = rgb
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    if brightness < threshold:
        return (255, 255, 255, 255)
    else:
        return (0, 0, 0, 255)


def _save_image(pil_img, out_base, formats, dpi=200, face_rgb01=(1, 1, 1)):
    saved_paths = []

    for fmt in formats:
        ext = str(fmt).lower().lstrip(".")
        out_path = f"{out_base}.{ext}"

        if ext in {"png", "jpg", "jpeg", "tif", "tiff"}:
            if ext in {"jpg", "jpeg"}:
                pil_img.convert("RGB").save(out_path, quality=95)
            else:
                pil_img.save(out_path, dpi=(dpi, dpi))
        else:
            _save_with_matplotlib(pil_img, out_base, [ext], dpi, face_rgb01)

        print(f"✅ Saved: {out_path}")
        saved_paths.append(out_path)

    return saved_paths


def _infer_column_kind(values):
    """
    Fallback inference for whether values are categorical or numerical.
    """
    arr = np.asarray(values)

    if arr.ndim > 1:
        return "numerical"

    if arr.dtype.kind in {"O", "U", "S"}:
        return "categorical"

    if np.issubdtype(arr.dtype, np.number):
        return "numerical"

    return "categorical"


def _resolve_column_kind(user_kind, values, column_name="value"):
    """
    Resolve whether the column should be treated as categorical or numerical.
    """
    if user_kind is not None:
        kind = str(user_kind).strip().lower()
        if kind not in {"categorical", "numerical"}:
            raise ValueError(
                f"column_kind must be 'categorical' or 'numerical', got {user_kind!r}"
            )
        print(f"  Using user-specified column type for {column_name!r}: {kind}")
        return kind

    inferred = _infer_column_kind(values)
    print(f"  Inferred column type for {column_name!r}: {inferred}")
    return inferred


def _load_values_from_h5ad(
    minimal_h5ad_path,
    key,
    vis_basis="spatial",
    value_index=None,
):
    """
    Load coordinates plus values from h5ad.

    Supports:
      - ad.obs[key] (1D)
      - ad.obsm[key] (1D or 2D)

    For 2D numerical arrays in obsm, user can specify value_index.
    """
    if not os.path.isfile(minimal_h5ad_path):
        raise FileNotFoundError(f"Clustering h5ad not found: {minimal_h5ad_path}")

    ad = sc.read_h5ad(minimal_h5ad_path)

    if vis_basis not in ad.obsm_keys():
        raise KeyError(f"{vis_basis!r} not found in ad.obsm.")

    coords_raw = ad.obsm[vis_basis].astype(np.float32)

    key_in_obs = key in ad.obs_keys()
    key_in_obsm = key in ad.obsm_keys()

    if not key_in_obs and not key_in_obsm:
        raise KeyError(f"{key!r} not found in ad.obs or ad.obsm.")

    if key_in_obs:
        values = ad.obs[key].to_numpy()
        source = "obs"
    else:
        values = np.asarray(ad.obsm[key])
        source = "obsm"

        if values.ndim == 2:
            if value_index is None:
                raise ValueError(
                    f"{key!r} is 2D in ad.obsm with shape {values.shape}. "
                    "Please provide value_index to choose which column to plot."
                )
            if not (0 <= int(value_index) < values.shape[1]):
                raise IndexError(
                    f"value_index={value_index} out of bounds for {key!r} with shape {values.shape}"
                )
            values = values[:, int(value_index)]

    del ad
    gc.collect()

    return coords_raw, np.asarray(values), source


def _make_numeric_colorbar_image(
    cmap_name="viridis",
    vmin=0.0,
    vmax=1.0,
    label="value",
    width=1200,
    height=180,
    dpi=200,
    facecolor="white",
    text_color="black",
):
    """
    Create a standalone horizontal colorbar as a PIL image.
    Sized generously to avoid clipping.
    """
    fig_w = width / float(dpi)
    fig_h = height / float(dpi)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=facecolor)
    ax = fig.add_axes([0.08, 0.45, 0.84, 0.22])

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap_name))
    sm.set_array([])

    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(label, color=text_color)
    cb.ax.xaxis.set_tick_params(color=text_color, labelcolor=text_color)
    cb.outline.set_edgecolor(text_color)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")

    plt.close(fig)
    return img


# ---------------------------------------------------------------------
# 1. Stitch HoVer-Net JSON tiles -> whole-slide coordinates
# ---------------------------------------------------------------------

_TILE_RE = re.compile(r"y(\d+)_x(\d+)", re.IGNORECASE)


def _parse_tile_origin_from_filename(fname):
    """
    Given '.../y00001280_x00001536.json', return (tile_y0, tile_x0) as ints.

    Matches the mask stitching convention:
        y0 = vertical offset (top row of tile)
        x0 = horizontal offset (left column of tile)
    """
    base = os.path.basename(fname)
    m = _TILE_RE.search(base)
    if not m:
        raise ValueError(f"Cannot parse tile origin from filename: {fname}")
    tile_y0 = int(m.group(1))
    tile_x0 = int(m.group(2))
    return tile_y0, tile_x0


def load_hovernet_nuclei(json_dir):
    """
    Load all HoVer-Net tile JSONs in `json_dir` and stitch coordinates
    back into whole-slide image space.

    Conventions:
      - Filenames: yYYYY_xXXXX.json
            * YYYY = top (row, y) of tile in WSI space
            * XXXX = left (column, x) of tile in WSI space
      - Within JSON per nucleus:
            * 'centroid': [x, y] in tile-local coordinates
            * 'contour':  [[x, y], ...] in tile-local coordinates

      - Global mapping:
            Y_global = tile_y0 + y_local
            X_global = tile_x0 + x_local

    Returns
    -------
    centroids_yx : (N, 2) float32
        Global centroids (y, x).
    contours_matrix : (P, 3) float32
        Rows: (nucleus_id, y, x).
    """
    json_paths = sorted(glob.glob(os.path.join(json_dir, "y*_x*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No HoVer-Net JSON tiles found in {json_dir}")

    centroids_list = []
    contours_list = []

    for jp in json_paths:
        tile_y0, tile_x0 = _parse_tile_origin_from_filename(jp)

        with open(jp, "r") as f:
            data = json.load(f)

        nuc_dict = data.get("nuc", {})

        for k, ninfo in sorted(
            nuc_dict.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0],
        ):
            centroid = ninfo.get("centroid", None)
            contour = ninfo.get("contour", None)
            if centroid is None or contour is None:
                continue

            x_local = float(centroid[0])
            y_local = float(centroid[1])

            Xg = tile_x0 + x_local
            Yg = tile_y0 + y_local

            nuc_id = len(centroids_list)
            centroids_list.append([Yg, Xg])

            for pt in contour:
                x_l = float(pt[0])
                y_l = float(pt[1])

                Xp = tile_x0 + x_l
                Yp = tile_y0 + y_l

                contours_list.append([nuc_id, Yp, Xp])

    centroids_yx = np.asarray(centroids_list, dtype=np.float32)
    contours_matrix = np.asarray(contours_list, dtype=np.float32)

    return centroids_yx, contours_matrix


def contours_matrix_to_polygons(contours_matrix, n_nuclei):
    """
    Convert contours_matrix (nucleus_id, y, x) to a list of polygons.

    Returns
    -------
    polygons_yx : list of (Mi, 2) arrays
        polygons_yx[i] has rows (y, x) for nucleus i.
    """
    polygons = [[] for _ in range(n_nuclei)]
    for nid, y, x in contours_matrix:
        nid_int = int(nid)
        if 0 <= nid_int < n_nuclei:
            polygons[nid_int].append((float(y), float(x)))

    polygons_yx = [
        np.asarray(poly, dtype=np.float32) if poly else np.empty((0, 2), dtype=np.float32)
        for poly in polygons
    ]
    return polygons_yx


# ---------------------------------------------------------------------
# 2. Plot values using nuclei/cell contours as masks (single sample)
# ---------------------------------------------------------------------


def plot_numerical_values_on_cell_masks(
    sample,
    he_path,
    hovernet_json_dir,
    save_dir,
    value_key,
    minimal_h5ad_path=None,
    coords=None,
    values=None,
    value_index=None,
    column_kind="numerical",
    vis_basis="spatial",
    spatial_scale_factor=16.0,
    max_match_dist_px=None,
    downsample_factor=1.0,
    background_color=(0, 0, 0),
    cmap_name="viridis",
    vmin=None,
    vmax=None,
    colorbar_label=None,
    save_colorbar=True,
    plot_title=None,
    out_formats=("png",),
    dpi=200,
):
    """
    Plot numerical values on cell masks using a continuous colormap.

    Typical use cases:
      - continuous score in ad.obs
      - one selected column from a 2D matrix in ad.obsm (using value_index)
      - probabilities such as ad.obsm['cdan_probs'][:, class_idx]
    """
    if not (0 < downsample_factor <= 1.0):
        raise ValueError(
            f"downsample_factor must be in (0, 1], got {downsample_factor}."
        )

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Processing numerical overlay for sample: {sample} ===")
    print(f"  H&E path:        {he_path}")
    print(f"  JSON dir:        {hovernet_json_dir}")
    print(f"  Save dir:        {save_dir}")
    print(f"  Value key:       {value_key}")

    if not os.path.isfile(he_path):
        raise FileNotFoundError(f"H&E image not found: {he_path}")
    if not os.path.isdir(hovernet_json_dir):
        raise FileNotFoundError(f"HoVer-Net JSON dir not found: {hovernet_json_dir}")

    he_img = pyvips.Image.new_from_file(he_path, access="sequential")
    if he_img.bands > 3:
        he_img = he_img[0:3]
    elif he_img.bands == 1:
        he_img = he_img.bandjoin([he_img, he_img, he_img])

    H, W = he_img.height, he_img.width
    print(f"  Full-resolution H&E size: {W} x {H}")

    if coords is not None and values is not None:
        coords_raw = np.asarray(coords, dtype=np.float32)
        values_arr = np.asarray(values)
        source = "direct"
    elif minimal_h5ad_path is not None:
        coords_raw, values_arr, source = _load_values_from_h5ad(
            minimal_h5ad_path=minimal_h5ad_path,
            key=value_key,
            vis_basis=vis_basis,
            value_index=value_index,
        )
    else:
        raise ValueError("Provide either minimal_h5ad_path OR coords+values.")

    resolved_kind = _resolve_column_kind(column_kind, values_arr, column_name=value_key)
    if resolved_kind != "numerical":
        raise ValueError(
            f"{value_key!r} resolved to {resolved_kind!r}, but "
            "plot_numerical_values_on_cell_masks requires numerical values."
        )

    values_arr = np.asarray(values_arr, dtype=np.float32)

    coords_yx_full = np.stack(
        [coords_raw[:, 1], coords_raw[:, 0]],
        axis=1,
    ) * float(spatial_scale_factor)

    print(f"  Loaded values from: {source}")
    print(f"  Value array shape: {values_arr.shape}")
    print(f"  Value dtype:       {values_arr.dtype}")

    print(f"  Loading HoVer-Net nuclei/cells from: {hovernet_json_dir}")
    nuc_centroids_yx, nuc_contours_matrix = load_hovernet_nuclei(hovernet_json_dir)
    n_nuclei = nuc_centroids_yx.shape[0]
    print(f"  Nuclei / cells count: {n_nuclei}")

    nuc_polygons_yx = contours_matrix_to_polygons(nuc_contours_matrix, n_nuclei)
    del nuc_contours_matrix
    gc.collect()

    print("  Building KDTree for clustering coords...")
    tree = cKDTree(coords_yx_full)
    print("  Querying nearest numerical values for nuclei centroids...")
    dists, idxs = tree.query(nuc_centroids_yx, k=1)

    nucleus_values = np.full(n_nuclei, np.nan, dtype=np.float32)
    for i in range(n_nuclei):
        if max_match_dist_px is None or dists[i] <= max_match_dist_px:
            nucleus_values[i] = values_arr[idxs[i]]

    del coords_raw
    del coords_yx_full
    del values_arr
    del tree
    del dists
    del idxs
    del nuc_centroids_yx
    gc.collect()

    valid_mask = np.isfinite(nucleus_values)
    if not np.any(valid_mask):
        print("  No nuclei could be matched to numerical values (skipping drawing).")
        return []

    if vmin is None:
        vmin = float(np.nanmin(nucleus_values))
    if vmax is None:
        vmax = float(np.nanmax(nucleus_values))
    if vmax <= vmin:
        vmax = vmin + 1e-8

    print(f"  Numerical scale: [{vmin}, {vmax}] using cmap={cmap_name}")

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)

    bg_rgb = _to_rgb_u8(background_color)
    bg_rgb01 = _to_rgb01(tuple(int(x) for x in bg_rgb))

    draw_scale = float(downsample_factor)
    vis_W = max(1, int(round(W * draw_scale)))
    vis_H = max(1, int(round(H * draw_scale)))

    print(f"  Drawing nuclei / cell polygons at display resolution: {vis_W} x {vis_H}")
    overlay_img = Image.new("RGB", (vis_W, vis_H), tuple(int(x) for x in bg_rgb.tolist()))
    draw = ImageDraw.Draw(overlay_img, "RGB")

    for nid in range(n_nuclei):
        val = nucleus_values[nid]
        if not np.isfinite(val):
            continue

        poly_yx = nuc_polygons_yx[nid]
        if poly_yx.size == 0:
            continue

        poly_xy = _scale_polygon_xy(poly_yx, draw_scale)
        if len(poly_xy) < 3:
            continue

        rgba = cmap(norm(float(val)))
        col = tuple(int(round(255 * c)) for c in rgba[:3])
        draw.polygon(poly_xy, fill=col)

    suffix = _slug(value_key)
    if value_index is not None:
        suffix = f"{suffix}_idx{int(value_index)}"

    sample_out_dir = os.path.join(save_dir, str(sample), suffix)
    os.makedirs(sample_out_dir, exist_ok=True)

    title_label = colorbar_label if colorbar_label is not None else value_key
    title_text = plot_title if plot_title is not None else f"{value_key}_{title_label}"

    title_text_rgb = _auto_text_color_for_bg(background_color)[:3]
    bg_rgb_tuple = tuple(int(x) for x in bg_rgb.tolist())

    combined_img = overlay_img

    if save_colorbar:
        cb_label = colorbar_label if colorbar_label is not None else value_key
        cb_img = _make_numeric_colorbar_image(
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax,
            label=cb_label,
            width=max(900, overlay_img.width),
            height=180,
            dpi=dpi,
            facecolor="black" if title_text_rgb == (255, 255, 255) else "white",
            text_color="white" if title_text_rgb == (255, 255, 255) else "black",
        )
        combined_img = _stack_images_vertical(
            combined_img,
            cb_img,
            background_color=bg_rgb_tuple,
            gap_px=max(12, overlay_img.height // 100),
            center=True,
        )

    combined_img = _add_title_above_image(
        combined_img,
        title=title_text,
        background_color=bg_rgb_tuple,
        text_color=title_text_rgb,
    )

    safe_name = _slug(title_text)
    out_base = os.path.join(sample_out_dir, safe_name)

    saved_paths = _save_image(
        combined_img,
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=bg_rgb01,
    )

    del he_img
    del overlay_img
    if "combined_img" in locals():
        del combined_img
    del nuc_polygons_yx
    del nucleus_values
    gc.collect()
    return saved_paths


def plot_values_on_cell_masks(
    sample,
    he_path,
    hovernet_json_dir,
    save_dir,
    value_key,
    minimal_h5ad_path=None,
    coords=None,
    values=None,
    labels=None,
    value_index=None,
    column_kind=None,
    vis_basis="spatial",
    spatial_scale_factor=16.0,
    max_match_dist_px=None,
    downsample_factor=1.0,
    background_color=(0, 0, 0),

    # categorical options
    label_to_color=None,
    legend_font_rel=0.025,
    legend_min_font_px=12,
    plot_title=None,

    # numerical options
    cmap_name="viridis",
    vmin=None,
    vmax=None,
    colorbar_label=None,
    save_colorbar=True,

    out_formats=("png",),
    dpi=200,
):
    """
    Dispatcher that chooses categorical or numerical plotting based on user input,
    with fallback inference only if column_kind is None.
    """
    if coords is None and minimal_h5ad_path is not None:
        coords_raw, raw_values, _source = _load_values_from_h5ad(
            minimal_h5ad_path=minimal_h5ad_path,
            key=value_key,
            vis_basis=vis_basis,
            value_index=value_index,
        )
        del coords_raw
        gc.collect()
    elif values is not None:
        raw_values = values
    elif labels is not None:
        raw_values = labels
    else:
        raise ValueError(
            "For dispatcher usage, provide minimal_h5ad_path or raw values/labels."
        )

    resolved_kind = _resolve_column_kind(column_kind, raw_values, column_name=value_key)

    if resolved_kind == "categorical":
        return plot_clusters_on_cell_masks(
            sample=sample,
            he_path=he_path,
            hovernet_json_dir=hovernet_json_dir,
            save_dir=save_dir,
            minimal_h5ad_path=minimal_h5ad_path,
            coords=coords,
            labels=labels if labels is not None else raw_values,
            cluster_key=value_key,
            vis_basis=vis_basis,
            spatial_scale_factor=spatial_scale_factor,
            max_match_dist_px=max_match_dist_px,
            downsample_factor=downsample_factor,
            background_color=background_color,
            label_to_color=label_to_color,
            legend_font_rel=legend_font_rel,
            legend_min_font_px=legend_min_font_px,
            plot_title=plot_title if plot_title is not None else str(value_key),
            out_formats=out_formats,
            dpi=dpi,
        )

    return plot_numerical_values_on_cell_masks(
        sample=sample,
        he_path=he_path,
        hovernet_json_dir=hovernet_json_dir,
        save_dir=save_dir,
        value_key=value_key,
        minimal_h5ad_path=minimal_h5ad_path,
        coords=coords,
        values=values if values is not None else raw_values,
        value_index=value_index,
        column_kind=resolved_kind,
        vis_basis=vis_basis,
        spatial_scale_factor=spatial_scale_factor,
        max_match_dist_px=max_match_dist_px,
        downsample_factor=downsample_factor,
        background_color=background_color,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        save_colorbar=save_colorbar,
        plot_title=plot_title,
        out_formats=out_formats,
        dpi=dpi,
    )


def plot_clusters_on_cell_masks(
    sample,
    he_path,
    hovernet_json_dir,
    save_dir,
    minimal_h5ad_path=None,
    coords=None,
    labels=None,
    cluster_key="hier_kmeans",
    vis_basis="spatial",
    spatial_scale_factor=16.0,
    max_match_dist_px=None,
    downsample_factor=1.0,
    background_color=(0, 0, 0),
    label_to_color=None,
    legend_font_rel=0.025,
    legend_min_font_px=12,
    plot_title=None,
    out_formats=("png",),
    dpi=200,
):
    """
    Plot categorical labels on cell masks.
    """
    if not (0 < downsample_factor <= 1.0):
        raise ValueError(
            f"downsample_factor must be in (0, 1], got {downsample_factor}. "
            "Set e.g. 0.25 for 4x downsample or 1.0 for no downsampling."
        )

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Processing sample: {sample} ===")
    print(f"  H&E path:        {he_path}")
    print(f"  JSON dir:        {hovernet_json_dir}")
    print(f"  Save dir:        {save_dir}")
    if minimal_h5ad_path is not None:
        print(f"  Clustering h5ad: {minimal_h5ad_path}")
    else:
        print("  Using coords+labels arrays directly (no h5ad).")

    if not os.path.isfile(he_path):
        raise FileNotFoundError(f"H&E image not found: {he_path}")
    if not os.path.isdir(hovernet_json_dir):
        raise FileNotFoundError(f"HoVer-Net JSON dir not found: {hovernet_json_dir}")

    he_img = pyvips.Image.new_from_file(he_path, access="sequential")

    if he_img.bands > 3:
        he_img = he_img[0:3]
    elif he_img.bands == 1:
        he_img = he_img.bandjoin([he_img, he_img, he_img])

    H, W = he_img.height, he_img.width
    print(f"  Full-resolution H&E size: {W} x {H}")

    if coords is not None and labels is not None:
        coords_raw = np.asarray(coords, dtype=np.float32)
        labels = np.asarray(labels)
    elif minimal_h5ad_path is not None:
        if not os.path.isfile(minimal_h5ad_path):
            raise FileNotFoundError(f"Clustering h5ad not found: {minimal_h5ad_path}")

        ad = sc.read_h5ad(minimal_h5ad_path)
        cluster_key_in_obs = cluster_key in ad.obs_keys()
        cluster_key_in_obsm = cluster_key in ad.obsm_keys()
        if vis_basis not in ad.obsm_keys():
            raise KeyError(f"{vis_basis!r} not found in ad.obsm.")
        if cluster_key not in ad.obs_keys():
            if cluster_key not in ad.obsm_keys():
                raise KeyError(f"{cluster_key!r} not found in ad.obs.")

        coords_raw = ad.obsm[vis_basis].astype(np.float32)
        if cluster_key_in_obs:
            labels = ad.obs[cluster_key].to_numpy()
        elif cluster_key_in_obsm:
            labels = ad.obsm[cluster_key].flatten()
        del ad
    else:
        raise ValueError("Provide either minimal_h5ad_path OR coords+labels.")

    coords_yx_full = np.stack(
        [coords_raw[:, 1], coords_raw[:, 0]],
        axis=1,
    ) * float(spatial_scale_factor)

    coords_yx_scaled = coords_yx_full

    print(f"  Loading HoVer-Net nuclei/cells from: {hovernet_json_dir}")
    nuc_centroids_yx, nuc_contours_matrix = load_hovernet_nuclei(hovernet_json_dir)
    n_nuclei = nuc_centroids_yx.shape[0]
    print(f"  Nuclei / cells count: {n_nuclei}")

    nuc_centroids_scaled = nuc_centroids_yx
    nuc_polygons_yx = contours_matrix_to_polygons(nuc_contours_matrix, n_nuclei)
    del nuc_contours_matrix
    gc.collect()

    print("  Building KDTree for clustering coords...")
    tree = cKDTree(coords_yx_scaled)
    print("  Querying nearest clusters for nuclei centroids...")
    dists, idxs = tree.query(nuc_centroids_scaled, k=1)

    nucleus_labels = []
    for i in range(n_nuclei):
        if max_match_dist_px is not None and dists[i] > max_match_dist_px:
            nucleus_labels.append(None)
        else:
            nucleus_labels.append(labels[idxs[i]])

    label_vals = [lbl for lbl in nucleus_labels if lbl is not None]
    if not label_vals:
        print("  No nuclei could be matched to cluster coordinates (skipping drawing).")
        return []

    unique_labels = sorted(set(label_vals), key=str)

    label_to_color_final = {}
    if label_to_color is not None:
        for k, col in label_to_color.items():
            label_to_color_final[k] = tuple(_to_rgb_u8(col).tolist())

    del nuc_centroids_scaled
    del coords_yx_scaled
    del tree
    del dists
    del idxs
    del nuc_centroids_yx
    gc.collect()

    missing = [lbl for lbl in unique_labels if lbl not in label_to_color_final]
    if missing:
        cmap = cm.get_cmap("tab20")
        n_missing = len(missing)
        for i, lbl in enumerate(missing):
            rgb01 = cmap(i / max(1, n_missing))[:3]
            col_u8 = np.array([int(255 * c) for c in rgb01], dtype=np.uint8)
            label_to_color_final[lbl] = tuple(col_u8.tolist())

    legend_items = [(lbl, label_to_color_final[lbl]) for lbl in unique_labels]

    bg_rgb = _to_rgb_u8(background_color)
    bg_rgb01 = _to_rgb01(tuple(int(x) for x in bg_rgb))

    draw_scale = float(downsample_factor)
    vis_W = max(1, int(round(W * draw_scale)))
    vis_H = max(1, int(round(H * draw_scale)))

    print(f"  Drawing nuclei / cell polygons at display resolution: {vis_W} x {vis_H}")
    overlay_img = Image.new("RGB", (vis_W, vis_H), tuple(int(x) for x in bg_rgb.tolist()))
    draw = ImageDraw.Draw(overlay_img, "RGB")

    for nid in range(n_nuclei):
        lbl = nucleus_labels[nid]
        if lbl is None:
            continue

        poly_yx = nuc_polygons_yx[nid]
        if poly_yx.size == 0:
            continue

        poly_xy = _scale_polygon_xy(poly_yx, draw_scale)
        if len(poly_xy) < 3:
            continue

        col = label_to_color_final[lbl]
        draw.polygon(poly_xy, fill=(int(col[0]), int(col[1]), int(col[2])))

    legend_text_color = _auto_text_color_for_bg(background_color)

    final_img = _add_legend_outside_right_transparent(
        overlay_img,
        legend_items,
        legend_font_rel=legend_font_rel,
        legend_min_font_px=legend_min_font_px,
        text_color=legend_text_color,
        strip_background_color=tuple(int(x) for x in bg_rgb.tolist()),
    )

    title_text = plot_title if plot_title is not None else str(cluster_key)
    title_text_color = _auto_text_color_for_bg(background_color)[:3]
    final_img = _add_title_above_image(
        final_img,
        title=title_text,
        background_color=tuple(int(x) for x in bg_rgb.tolist()),
        text_color=title_text_color,
    )

    sample_out_dir = os.path.join(save_dir, str(sample), _slug(cluster_key))
    os.makedirs(sample_out_dir, exist_ok=True)
    out_base = os.path.join(sample_out_dir, _slug(title_text))

    saved_paths = _save_image(
        final_img,
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=bg_rgb01,
    )

    del he_img
    del overlay_img
    del final_img
    del nuc_polygons_yx
    del nucleus_labels
    gc.collect()
    return saved_paths


def plot_selected_cluster_mask_on_he(
    sample,
    he_path,
    hovernet_json_dir,
    save_dir,
    selected_cluster,
    minimal_h5ad_path=None,
    coords=None,
    labels=None,
    cluster_key="hier_kmeans",
    vis_basis="spatial",
    spatial_scale_factor=16.0,
    max_match_dist_px=None,

    downsample_factor=0.25,
    mode="masked",
    masked_background=(0, 0, 0),
    draw_boundaries=True,
    boundary_color=(255, 0, 0),
    boundary_width_px=2,

    fill_cells=False,
    fill_color=(255, 0, 0),
    fill_alpha=80,

    background_style="solid",
    outside_color=(0, 0, 0),
    fade_desaturate=0.85,
    fade_whiten=0.35,

    out_formats=("png",),
    dpi=200,
):
    """
    After assigning cluster labels to nuclei/cells, plot ONLY the selected cluster's
    cell mask on top of H&E.

    Modes
    -----
    mode="masked":
        Show ONLY H&E pixels covered by the selected cluster's cells.
        Everything else is `masked_background`.

    mode="boundaries":
        Show ALL H&E, and overlay cell boundaries (and optional translucent fill)
        for cells in the selected cluster.
    """
    if not (0 < downsample_factor <= 1.0):
        raise ValueError(f"downsample_factor must be in (0,1], got {downsample_factor}")
    if mode not in ("masked", "boundaries"):
        raise ValueError("mode must be either 'masked' or 'boundaries'")

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Processing sample (H&E overlay): {sample} ===")
    print(f"  H&E path:        {he_path}")
    print(f"  JSON dir:        {hovernet_json_dir}")
    print(f"  Save dir:        {save_dir}")
    print(f"  Selected cluster: {selected_cluster!r}")
    print(f"  Mode:            {mode}")
    if minimal_h5ad_path is not None:
        print(f"  Clustering h5ad: {minimal_h5ad_path}")
    else:
        print("  Using coords+labels arrays directly (no h5ad).")

    if not os.path.isfile(he_path):
        raise FileNotFoundError(f"H&E image not found: {he_path}")
    if not os.path.isdir(hovernet_json_dir):
        raise FileNotFoundError(f"HoVer-Net JSON dir not found: {hovernet_json_dir}")

    he_img = pyvips.Image.new_from_file(he_path, access="sequential")
    he_img = _ensure_vips_rgb_u8(he_img)
    H, W = he_img.height, he_img.width
    print(f"  Full-resolution H&E size: {W} x {H}")

    if coords is not None and labels is not None:
        coords_raw = np.asarray(coords, dtype=np.float32)
        labels_arr = np.asarray(labels)
    elif minimal_h5ad_path is not None:
        if not os.path.isfile(minimal_h5ad_path):
            raise FileNotFoundError(f"Clustering h5ad not found: {minimal_h5ad_path}")
        ad = sc.read_h5ad(minimal_h5ad_path)

        if vis_basis not in ad.obsm_keys():
            raise KeyError(f"{vis_basis!r} not found in ad.obsm.")
        if cluster_key not in ad.obs_keys():
            if cluster_key not in ad.obsm_keys():
                raise KeyError(f"{cluster_key!r} not found in ad.obs.")
        cluster_key_in_obs = cluster_key in ad.obs_keys()
        cluster_key_in_obsm = cluster_key in ad.obsm_keys()
        coords_raw = ad.obsm[vis_basis].astype(np.float32)
        if cluster_key_in_obs:
            labels_arr = ad.obs[cluster_key].to_numpy()
        elif cluster_key_in_obsm:
            labels_arr = ad.obsm[cluster_key].flatten()
        del ad
    else:
        raise ValueError("Provide either minimal_h5ad_path OR coords+labels.")

    coords_yx_full = np.stack([coords_raw[:, 1], coords_raw[:, 0]], axis=1) * float(spatial_scale_factor)

    print(f"  Loading HoVer-Net nuclei/cells from: {hovernet_json_dir}")
    nuc_centroids_yx, nuc_contours_matrix = load_hovernet_nuclei(hovernet_json_dir)
    n_nuclei = nuc_centroids_yx.shape[0]
    print(f"  Nuclei / cells count: {n_nuclei}")

    nuc_polygons_yx = contours_matrix_to_polygons(nuc_contours_matrix, n_nuclei)
    del nuc_contours_matrix
    gc.collect()

    print("  Building KDTree for clustering coords...")
    tree = cKDTree(coords_yx_full)
    print("  Querying nearest clusters for nuclei centroids...")
    dists, idxs = tree.query(nuc_centroids_yx, k=1)

    nucleus_labels = []
    for i in range(n_nuclei):
        if max_match_dist_px is not None and dists[i] > max_match_dist_px:
            nucleus_labels.append(None)
        else:
            nucleus_labels.append(labels_arr[idxs[i]])

    selected_nids = [
        nid for nid, lbl in enumerate(nucleus_labels)
        if (lbl is not None and _labels_equal(lbl, selected_cluster))
    ]
    if not selected_nids:
        print("  No nuclei matched the selected cluster (nothing to draw).")
        return []

    print(f"  Cells in selected cluster: {len(selected_nids)}")

    if downsample_factor < 1.0:
        he_vis = he_img.resize(downsample_factor)
    else:
        he_vis = he_img

    he_pil = _vips_to_pil_rgb(he_vis).convert("RGBA")
    vis_W, vis_H = he_pil.size
    print(f"  Display size (downsampled): {vis_W} x {vis_H}")

    scale = float(downsample_factor)

    mask = Image.new("L", (vis_W, vis_H), 0)
    mask_draw = ImageDraw.Draw(mask)

    overlay = Image.new("RGBA", (vis_W, vis_H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for nid in selected_nids:
        poly_yx = nuc_polygons_yx[nid]
        if poly_yx.size == 0:
            continue

        poly_xy = _scale_polygon_xy(poly_yx, scale)
        if len(poly_xy) < 3:
            continue

        mask_draw.polygon(poly_xy, fill=255)

        if mode == "boundaries":
            if fill_cells:
                overlay_draw.polygon(
                    poly_xy,
                    fill=(int(fill_color[0]), int(fill_color[1]), int(fill_color[2]), int(fill_alpha)),
                )
            if draw_boundaries:
                overlay_draw.line(
                    poly_xy + [poly_xy[0]],
                    fill=(int(boundary_color[0]), int(boundary_color[1]), int(boundary_color[2]), 255),
                    width=int(boundary_width_px),
                    joint="curve",
                )

    if mode == "masked":
        if background_style not in ("solid", "he_faded"):
            raise ValueError("background_style must be 'solid' or 'he_faded'")

        if background_style == "solid":
            bg = Image.new(
                "RGBA",
                (vis_W, vis_H),
                (int(outside_color[0]), int(outside_color[1]), int(outside_color[2]), 255),
            )
            outside = bg
            face_rgb01 = (outside_color[0] / 255.0, outside_color[1] / 255.0, outside_color[2] / 255.0)
        else:
            outside = _pil_desaturate_and_whiten(
                he_pil,
                desaturate=fade_desaturate,
                whiten=fade_whiten,
            )
            face_rgb01 = (1.0, 1.0, 1.0)

        composed = Image.composite(he_pil, outside, mask)
        out_name = f"he_masked_cluster_{_slug(selected_cluster)}_{background_style}"
    else:
        composed = Image.alpha_composite(he_pil, overlay)
        face_rgb01 = (1.0, 1.0, 1.0)
        out_name = f"he_boundaries_cluster_{_slug(selected_cluster)}"

    sample_out_dir = os.path.join(save_dir, str(sample), _slug(cluster_key))
    os.makedirs(sample_out_dir, exist_ok=True)
    out_base = os.path.join(sample_out_dir, out_name)

    saved_paths = _save_with_matplotlib(
        composed.convert("RGB"),
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=face_rgb01,
    )

    del he_img
    del he_pil
    del mask
    del overlay
    del composed
    del nuc_polygons_yx
    del nucleus_labels
    gc.collect()
    return saved_paths