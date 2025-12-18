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
from matplotlib import cm

from PIL import Image, ImageDraw, ImageFont

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
):
    """
    Add a transparent legend strip to the RIGHT of the image (top-right aligned).
    items: list[(label, (R,G,B))]
    """
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
        drawp.text((tx, ty), str(lbl), fill=(0, 0, 0, 255), font=font)
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
        if vis_basis not in ad.obsm_keys():
            raise KeyError(f"{vis_basis!r} not found in ad.obsm.")
        if cluster_key not in ad.obs_keys():
            raise KeyError(f"{cluster_key!r} not found in ad.obs.")

        coords_raw = ad.obsm[vis_basis].astype(np.float32)
        labels = ad.obs[cluster_key].to_numpy()
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

    print("  Drawing nuclei / cell polygons at full resolution...")
    overlay_img = Image.new("RGB", (W, H), tuple(int(x) for x in bg_rgb.tolist()))
    draw = ImageDraw.Draw(overlay_img, "RGB")

    for nid in range(n_nuclei):
        lbl = nucleus_labels[nid]
        if lbl is None:
            continue

        poly_yx = nuc_polygons_yx[nid]
        if poly_yx.size == 0:
            continue

        # convert (y, x) -> (x, y) for PIL
        poly_xy = [(float(x), float(y)) for (y, x) in poly_yx]

        col = label_to_color_final[lbl]
        draw.polygon(poly_xy, fill=(int(col[0]), int(col[1]), int(col[2])))

    # add legend to the right (transparent strip) at full res
    overlay_with_legend = _add_legend_outside_right_transparent(
        overlay_img,
        legend_items,
        legend_font_rel=legend_font_rel,
        legend_min_font_px=legend_min_font_px,
    )

    # --- FINAL DOWNSAMPLE (if requested) ---
    final_img = overlay_with_legend
    if downsample_factor < 1.0:
        W_full, H_full = final_img.size
        new_W = max(1, int(round(W_full * downsample_factor)))
        new_H = max(1, int(round(H_full * downsample_factor)))
        print(f"  Downsampling final image: {W_full}x{H_full} -> {new_W}x{new_H}")
        final_img = final_img.resize((new_W, new_H), resample=Image.BILINEAR)

    # --- save ---
    sample_out_dir = os.path.join(save_dir, str(sample), _slug(cluster_key))
    os.makedirs(sample_out_dir, exist_ok=True)
    out_base = os.path.join(sample_out_dir, "clusters_on_cell_masks")

    saved_paths = _save_with_matplotlib(
        final_img,
        out_base=out_base,
        formats=out_formats,
        dpi=dpi,
        face_rgb01=bg_rgb01,
    )

    del he_img
    gc.collect()
    return saved_paths
