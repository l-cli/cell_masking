#!/usr/bin/env python
"""
Utilities to stitch HoVer-Net JSON tiles into whole-slide coordinates and
plot clustering results restricted to nuclei / cell contours.

Core entrypoint:
    plot_clusters_on_cell_masks(...)

Typical use:
    from plot_clusters_on_cell_masks import plot_clusters_on_cell_masks

    plot_clusters_on_cell_masks(
        sample="MySample",
        he_path="/path/to/he_raw.tif",
        hovernet_json_dir="/path/to/hover_net_out/MySample/json_expanded",
        save_dir="/path/to/output/plots",
        minimal_h5ad_path="/path/to/minimal.h5ad",
        cluster_key="my_cluster_column",
        vis_basis="spatial",
        spatial_scale_factor=16.0,
        max_match_dist_px=16.0,
        label_to_color={...},
        downsample_factor=0.5,
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


def _add_legend_outside_right_transparent(
    pil_img,
    items,
    legend_font_rel=0.04,
    legend_min_font_px=16,
    text_color=None,
):
    """
    Add a transparent legend strip to the RIGHT of the image (top-right aligned).
    items: list[(label, (R,G,B))]
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
    box_pad = max(8, int(round(text_px * 0.6)))   # internal padding
    strip_pad = max(8, int(round(text_px * 0.6))) # from image edge

    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", text_px)
    except Exception:
        try:
            import matplotlib.font_manager as fm
            fpath = fm.findfont("DejaVu Sans", fallback_to_default=True)
            font = ImageFont.truetype(fpath, text_px)
        except Exception:
            font = ImageFont.load_default()

    draw_base = ImageDraw.Draw(base)

    def _text_size(txt):
        try:
            bbox = draw_base.textbbox((0, 0), str(txt), font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            return draw_base.textsize(str(txt), font=font)

    widths, heights = zip(*(_text_size(lbl) for lbl, _ in items)) if items else ([40], [text_px])
    text_h = max(heights) if heights else text_px
    row_h = max(swatch, text_h) + line_gap
    content_w = swatch + gap + (max(widths) if widths else 40)
    content_h = len(items) * row_h - line_gap

    legend_w = box_pad * 2 + content_w
    legend_h = box_pad * 2 + content_h

    # transparent canvas
    strip_w = strip_pad * 2 + legend_w
    new_W = W + strip_w
    new_H = max(H, strip_pad * 2 + legend_h)
    canvas = Image.new("RGBA", (new_W, new_H), (0, 0, 0, 0))
    canvas.alpha_composite(base, (0, 0))

    drawp = ImageDraw.Draw(canvas, "RGBA")

    # legend origin: top-right inside the transparent strip
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

    # Convert to uchar if needed
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
    # (y,x) -> (x,y), scaled
    return [(float(x) * scale, float(y) * scale) for (y, x) in poly_yx]


def _labels_equal(a, b) -> bool:
    """Robust label equality for mixed dtypes / categoricals."""
    # Most of the time plain equality is fine; fallback to string compare.
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

    # 1) desaturate
    # Color factor: 1 -> original, 0 -> grayscale
    color_factor = max(0.0, min(1.0, 1.0 - float(desaturate)))
    rgb_desat = ImageEnhance.Color(rgb).enhance(color_factor)

    # 2) blend toward white
    whiten = max(0.0, min(1.0, float(whiten)))
    white = Image.new("RGB", rgb_desat.size, (255, 255, 255))
    rgb_faded = Image.blend(rgb_desat, white, whiten)

    # restore alpha from original
    a = rgba_img.split()[-1]
    out = rgb_faded.convert("RGBA")
    out.putalpha(a)
    return out

def _auto_text_color_for_bg(bg_color, threshold=186):
    """
    Return white text for dark backgrounds, black text for light backgrounds.

    bg_color can be:
      - '#RRGGBB' or '#RGB'
      - (R,G,B)

    threshold:
      larger -> more likely to choose white text
    """
    rgb = _to_rgb_u8(bg_color).astype(float)
    r, g, b = rgb

    # perceived brightness
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    if brightness < threshold:
        return (255, 255, 255, 255)  # white
    else:
        return (0, 0, 0, 255)        # black

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
            # fallback to matplotlib only for vector or unusual formats
            _save_with_matplotlib(pil_img, out_base, [ext], dpi, face_rgb01)

        print(f"✅ Saved: {out_path}")
        saved_paths.append(out_path)

    return saved_paths
    

def _infer_column_kind(values):
    """
    Fallback inference for whether values are categorical or numerical.

    Returns
    -------
    kind : str
        "categorical" or "numerical"
    """
    arr = np.asarray(values)

    # multi-dimensional numeric arrays are treated as numerical
    if arr.ndim > 1:
        return "numerical"

    # object / string / category-like -> categorical
    if arr.dtype.kind in {"O", "U", "S"}:
        return "categorical"

    # numeric -> numerical
    if np.issubdtype(arr.dtype, np.number):
        return "numerical"

    # conservative fallback
    return "categorical"


def _resolve_column_kind(user_kind, values, column_name="value"):
    """
    Resolve whether the column should be treated as categorical or numerical.

    Parameters
    ----------
    user_kind : str or None
        User-specified kind. Expected: "categorical" or "numerical".
        If None, use fallback inference.
    values : array-like
        The raw values.
    column_name : str
        For messages.

    Returns
    -------
    kind : str
        "categorical" or "numerical"
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
    width=280,
    height=80,
    dpi=200,
    facecolor="white",
    text_color="black",
):
    """
    Create a standalone horizontal colorbar as a PIL image.
    """
    fig_w = width / float(dpi)
    fig_h = height / float(dpi)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap_name))
    sm.set_array([])

    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.6, pad=0.35)
    cb.set_label(label, color=text_color)
    cb.ax.xaxis.set_tick_params(color=text_color)
    plt.setp(cb.ax.get_xticklabels(), color=text_color)

    ax.remove()
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")

    plt.close(fig)
    return img


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
    column_kind="numerical",          # user-specified; fallback inference if None
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
    out_formats=("png",),
    dpi=200,
):
    """
    Plot numerical values on cell masks using a continuous colormap.

    Typical use cases:
      - continuous score in ad.obs
      - one selected column from a 2D matrix in ad.obsm (using value_index)
      - probabilities such as ad.obsm['cdan_probs'][:, class_idx]

    Parameters
    ----------
    value_key : str
        Column/key to plot from ad.obs or ad.obsm.
    value_index : int or None
        Required if value_key in ad.obsm is 2D and you want a specific column.
    column_kind : {"numerical", "categorical"} or None
        User-specified type. If None, a fallback inference is used.
        For this function, resolved type must be numerical.
    save_colorbar : bool
        Save a separate colorbar image.
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

    # --- values + coords ---
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

    out_base = os.path.join(sample_out_dir, "numerical_values_on_cell_masks")
    saved_paths = _save_image(
        overlay_img,
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=bg_rgb01,
    )

    if save_colorbar:
        label = colorbar_label if colorbar_label is not None else value_key
        if value_index is not None:
            label = f"{label} [col {int(value_index)}]"

        text_color = "white" if _auto_text_color_for_bg(background_color)[:3] == (255, 255, 255) else "black"
        facecolor = "black" if text_color == "white" else "white"

        cb_img = _make_numeric_colorbar_image(
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax,
            label=label,
            dpi=dpi,
            facecolor=facecolor,
            text_color=text_color,
        )
        cb_base = os.path.join(sample_out_dir, "numerical_values_colorbar")
        saved_paths.extend(
            _save_image(
                cb_img,
                out_base=cb_base,
                formats=out_formats,
                dpi=dpi,
                face_rgb01=(0, 0, 0) if facecolor == "black" else (1, 1, 1),
            )
        )

    del he_img
    del overlay_img
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
    column_kind=None,                 # "categorical" or "numerical", fallback inference if None
    vis_basis="spatial",
    spatial_scale_factor=16.0,
    max_match_dist_px=None,
    downsample_factor=1.0,
    background_color=(0, 0, 0),

    # categorical options
    label_to_color=None,
    legend_font_rel=0.025,
    legend_min_font_px=12,

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
        coords_raw, raw_values, source = _load_values_from_h5ad(
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
        out_formats=out_formats,
        dpi=dpi,
    )


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
    json_paths = sorted(
        glob.glob(os.path.join(json_dir, "y*_x*.json"))
    )
    if not json_paths:
        raise FileNotFoundError(f"No HoVer-Net JSON tiles found in {json_dir}")

    centroids_list = []
    contours_list = []

    for jp in json_paths:
        tile_y0, tile_x0 = _parse_tile_origin_from_filename(jp)

        with open(jp, "r") as f:
            data = json.load(f)

        nuc_dict = data.get("nuc", {})

        # stable ordering
        for k, ninfo in sorted(
            nuc_dict.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0],
        ):
            centroid = ninfo.get("centroid", None)
            contour = ninfo.get("contour", None)
            if centroid is None or contour is None:
                continue

            # centroid: [x, y] tile-local -> global (y, x)
            x_local = float(centroid[0])
            y_local = float(centroid[1])

            Xg = tile_x0 + x_local
            Yg = tile_y0 + y_local

            nuc_id = len(centroids_list)  # 0-based global index
            centroids_list.append([Yg, Xg])

            # contour: list of [x, y] tile-local
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
# 2. Plot clusters using nuclei/cell contours as masks (single sample)
# ---------------------------------------------------------------------


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
    spatial_scale_factor=16.0,   # e.g. Visium 'spatial' -> H&E pixels
    max_match_dist_px=None,      # nuclei farther than this from any cluster coord are skipped
    downsample_factor=1.0,       # only used to downsample the *final* image (0 < d <= 1)
    background_color=(0, 0, 0),
    label_to_color=None,         # optional dict {label -> (R,G,B) or "#RRGGBB"}
    legend_font_rel=0.025,
    legend_min_font_px=12,
    out_formats=("png",),
    dpi=200,
):
    """
    For one sample:
      1. Load H&E from `he_path` (pyvips) to get dimensions.
      2. Load nuclei/cell centroids & contours from `hovernet_json_dir`.
         (This can be *expanded* cell masks if you've run expand_hovernet_cells.py.)
      3. Load clustering coords & labels either from:
           - `minimal_h5ad_path` (AnnData with obsm[vis_basis] & obs[cluster_key]), or
           - `coords` + `labels` arrays (in the same coordinate system).
      4. Assign each nucleus/cell the cluster label of the nearest clustering coord
         (in matched pixel space).
      5. Draw polygons filled by cluster color on a black background.
      6. Save to:
           save_dir/<sample>/<cluster_key>/clusters_on_cell_masks.{png,pdf,...}

    Returns
    -------
    saved_paths : list of str
        Paths to all saved figure files.
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

    # --- H&E for FULL-RES dimensions only ---
    he_img = pyvips.Image.new_from_file(he_path, access="sequential")

    # enforce 3 bands
    if he_img.bands > 3:
        he_img = he_img[0:3]
    elif he_img.bands == 1:
        he_img = he_img.bandjoin([he_img, he_img, he_img])

    # IMPORTANT: no resize here; we keep full-res dimensions
    H, W = he_img.height, he_img.width
    print(f"  Full-resolution H&E size: {W} x {H}")

    # --- clustering coords & labels ---
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

    # coords_raw ~ (x, y) -> FULL-RES (y, x) in H&E pixels
    coords_yx_full = np.stack(
        [coords_raw[:, 1], coords_raw[:, 0]],
        axis=1,
    ) * float(spatial_scale_factor)

    coords_yx_scaled = coords_yx_full  # no extra scaling here

    # --- nuclei / cells from HoVer-Net JSONs ---
    print(f"  Loading HoVer-Net nuclei/cells from: {hovernet_json_dir}")
    nuc_centroids_yx, nuc_contours_matrix = load_hovernet_nuclei(hovernet_json_dir)
    n_nuclei = nuc_centroids_yx.shape[0]
    print(f"  Nuclei / cells count: {n_nuclei}")

    nuc_centroids_scaled = nuc_centroids_yx  # already in full-res coords
    nuc_polygons_yx = contours_matrix_to_polygons(
        nuc_contours_matrix, n_nuclei
    )

    # --- nearest neighbor: assign cluster label to each nucleus centroid ---
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

    # --- build color mapping ---
    label_vals = [lbl for lbl in nucleus_labels if lbl is not None]
    if not label_vals:
        print("  No nuclei could be matched to cluster coordinates (skipping drawing).")
        return []

    unique_labels = sorted(set(label_vals), key=str)

    label_to_color_final = {}
    if label_to_color is not None:
        for k, col in label_to_color.items():
            label_to_color_final[k] = tuple(_to_rgb_u8(col).tolist())
    
    # free memory from large arrays we no longer need before drawing
    del nuc_centroids_scaled
    del coords_yx_scaled
    del tree
    del dists
    del idxs
    gc.collect()
    
    # assign colors for any missing labels
    missing = [lbl for lbl in unique_labels if lbl not in label_to_color_final]
    if missing:
        cmap = cm.get_cmap("tab20")
        n_missing = len(missing)
        for i, lbl in enumerate(missing):
            rgb01 = cmap(i / max(1, n_missing))[:3]
            col_u8 = np.array([int(255 * c) for c in rgb01], dtype=np.uint8)
            label_to_color_final[lbl] = tuple(col_u8.tolist())

    legend_items = [(lbl, label_to_color_final[lbl]) for lbl in unique_labels]

    # --- draw polygons on black at FULL resolution ---
    bg_rgb = _to_rgb_u8(background_color)
    bg_rgb01 = _to_rgb01(tuple(int(x) for x in bg_rgb))

    # draw directly at output resolution
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
    )

    # --- save ---
    sample_out_dir = os.path.join(save_dir, str(sample), _slug(cluster_key))
    os.makedirs(sample_out_dir, exist_ok=True)
    out_base = os.path.join(sample_out_dir, "clusters_on_cell_masks")

    saved_paths = _save_image(
        final_img,
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=bg_rgb01,
    )

    del he_img
    gc.collect()
    return saved_paths


def plot_selected_cluster_mask_on_he(
    sample,
    he_path,
    hovernet_json_dir,
    save_dir,
    selected_cluster,                 # <-- user-selected cluster label/value
    minimal_h5ad_path=None,
    coords=None,
    labels=None,
    cluster_key="hier_kmeans",
    vis_basis="spatial",
    spatial_scale_factor=16.0,
    max_match_dist_px=None,

    # ---- display controls ----
    downsample_factor=0.25,           # strongly recommended < 1 for WSI
    mode="masked",                    # "masked" or "boundaries"
    masked_background=(0, 0, 0),      # background for masked mode (outside cluster cells)
    draw_boundaries=True,             # relevant for mode="boundaries"
    boundary_color=(255, 0, 0),
    boundary_width_px=2,

    # optional translucent fill on top of H&E (useful in boundaries mode)
    fill_cells=False,
    fill_color=(255, 0, 0),
    fill_alpha=80,                    # 0-255

    # H&E background style (only for masked mode; boundaries mode always shows full H&E):
    background_style="solid",          # "solid" or "he_faded"
    outside_color=(0, 0, 0),           # used if background_style="solid"
    fade_desaturate=0.85,              # used if background_style="he_faded"
    fade_whiten=0.35,                  # used if background_style="he_faded"

    # ---- saving ----
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

    Returns
    -------
    saved_paths : list[str]
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

    # --- Read H&E with pyvips (full-res for dimensions; we will display downsampled) ---
    he_img = pyvips.Image.new_from_file(he_path, access="sequential")
    he_img = _ensure_vips_rgb_u8(he_img)
    H, W = he_img.height, he_img.width
    print(f"  Full-resolution H&E size: {W} x {H}")

    # --- clustering coords & labels ---
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

    # coords_raw ~ (x, y) -> full-res (y, x) in H&E pixels
    coords_yx_full = np.stack([coords_raw[:, 1], coords_raw[:, 0]], axis=1) * float(spatial_scale_factor)

    # --- nuclei / cells from HoVer-Net JSONs ---
    print(f"  Loading HoVer-Net nuclei/cells from: {hovernet_json_dir}")
    nuc_centroids_yx, nuc_contours_matrix = load_hovernet_nuclei(hovernet_json_dir)
    n_nuclei = nuc_centroids_yx.shape[0]
    print(f"  Nuclei / cells count: {n_nuclei}")

    nuc_polygons_yx = contours_matrix_to_polygons(nuc_contours_matrix, n_nuclei)

    # --- nearest neighbor: assign cluster label to each nucleus centroid ---
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

    # --- collect polygons for the selected cluster ---
    selected_nids = [
        nid for nid, lbl in enumerate(nucleus_labels)
        if (lbl is not None and _labels_equal(lbl, selected_cluster))
    ]
    if not selected_nids:
        print("  No nuclei matched the selected cluster (nothing to draw).")
        return []

    print(f"  Cells in selected cluster: {len(selected_nids)}")

    # --- build a downsampled H&E PIL for visualization ---
    if downsample_factor < 1.0:
        he_vis = he_img.resize(downsample_factor)
    else:
        he_vis = he_img

    he_pil = _vips_to_pil_rgb(he_vis).convert("RGBA")
    vis_W, vis_H = he_pil.size
    print(f"  Display size (downsampled): {vis_W} x {vis_H}")

    # Scale factor from full-res polygons to display pixels
    scale = float(downsample_factor)

    # --- create a mask image for selected cluster (in display resolution) ---
    mask = Image.new("L", (vis_W, vis_H), 0)
    mask_draw = ImageDraw.Draw(mask)

    # For boundaries mode, prepare an overlay RGBA for lines/fills
    overlay = Image.new("RGBA", (vis_W, vis_H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for nid in selected_nids:
        poly_yx = nuc_polygons_yx[nid]
        if poly_yx.size == 0:
            continue

        poly_xy = _scale_polygon_xy(poly_yx, scale)
        if len(poly_xy) < 3:
            continue

        # Fill mask (used in masked mode, and also optional fill in boundaries mode)
        mask_draw.polygon(poly_xy, fill=255)

        if mode == "boundaries":
            if fill_cells:
                overlay_draw.polygon(
                    poly_xy,
                    fill=(int(fill_color[0]), int(fill_color[1]), int(fill_color[2]), int(fill_alpha)),
                )
            if draw_boundaries:
                # Draw polygon outline
                overlay_draw.line(
                    poly_xy + [poly_xy[0]],
                    fill=(int(boundary_color[0]), int(boundary_color[1]), int(boundary_color[2]), 255),
                    width=int(boundary_width_px),
                    joint="curve",
                )

    # --- compose final image ---
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

        else:  # "he_faded"
            outside = _pil_desaturate_and_whiten(
                he_pil,
                desaturate=fade_desaturate,
                whiten=fade_whiten,
            )
            # background is still H&E-ish; white is usually a nice figure canvas
            face_rgb01 = (1.0, 1.0, 1.0)

        # Inside mask: original H&E; Outside mask: chosen background
        composed = Image.composite(he_pil, outside, mask)

        out_name = f"he_masked_cluster_{_slug(selected_cluster)}_{background_style}"
    else:
        # boundaries: show all H&E + overlay
        composed = Image.alpha_composite(he_pil, overlay)
        face_rgb01 = (1.0, 1.0, 1.0)
        out_name = f"he_boundaries_cluster_{_slug(selected_cluster)}"

    # --- save ---
    sample_out_dir = os.path.join(save_dir, str(sample), _slug(cluster_key))
    os.makedirs(sample_out_dir, exist_ok=True)
    out_base = os.path.join(sample_out_dir, out_name)

    # save with matplotlib for pdf/svg support
    saved_paths = _save_with_matplotlib(
        composed.convert("RGB"),
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=face_rgb01,
    )

    del he_img
    gc.collect()
    return saved_paths