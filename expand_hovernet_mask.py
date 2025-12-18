#!/usr/bin/env python
"""
Expand HoVer-Net nuclei contours into approximate whole-cell masks using
skimage.segmentation.expand_labels, and save back to JSON tiles.

Input JSON format (per tile) is like:
{
    "mag": null,
    "nuc": {
        "1": {
            "bbox": [[ymin, xmin], [ymax, xmax]],
            "centroid": [x_center, y_center],
            "contour": [[x0, y0], [x1, y1], ...],
            "type_prob": ...,
            "type": ...
        },
        "2": { ... },
        ...
    }
}

This script:
  - builds a label image for each tile from 'contour' polygons,
  - expands labels using skimage.segmentation.expand_labels(distance=D),
  - extracts new contours / centroids / bboxes for each label,
  - writes new JSONs with the *same structure*, but with
        'contour', 'centroid', and 'bbox' corresponding to the
        expanded (cell-sized) regions.

Usage:
    python expand_hovernet_cells.py \
        --in-json-dir  /path/to/hover_net_out/SAMPLE/json \
        --out-json-dir /path/to/hover_net_out/SAMPLE/json_cells \
        --distance-px  8

You can play with --distance-px to control how far to expand the labels.
"""

import os
import json
import glob
import argparse

import numpy as np

from skimage.draw import polygon as draw_polygon
from skimage.segmentation import expand_labels
from skimage.measure import regionprops, find_contours


def build_label_image_from_nuc_dict(nuc_dict):
    """
    Build a label image from HoVer-Net 'nuc' dict.

    Parameters
    ----------
    nuc_dict : dict
        data["nuc"] from one tile JSON.

    Returns
    -------
    labels : (H, W) int32 ndarray
        Label image (0 = background, 1..N = nuclei).
    key_order : list of str
        List of nucleus keys in the order they were mapped to label IDs.
        labels == i corresponds to nucleus key key_order[i-1].
    """
    if not nuc_dict:
        return None, []

    # Determine tile size from contours and bboxes
    max_x = 0
    max_y = 0

    for k, ninfo in nuc_dict.items():
        contour = ninfo.get("contour", None)
        bbox = ninfo.get("bbox", None)

        if contour is not None:
            for x, y in contour:
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

        if bbox is not None:
            # bbox: [[ymin, xmin], [ymax, xmax]]
            (ymin, xmin), (ymax, xmax) = bbox
            max_x = max(max_x, xmax)
            max_y = max(max_y, ymax)

    # +1 because coords are 0-based indices
    H = int(np.ceil(max_y)) + 1
    W = int(np.ceil(max_x)) + 1

    if H <= 0 or W <= 0:
        return None, []

    labels = np.zeros((H, W), dtype=np.int32)

    # Stable ordering of keys (numeric if possible)
    key_order = sorted(
        nuc_dict.keys(),
        key=lambda kk: int(kk) if str(kk).isdigit() else kk
    )

    for i, k in enumerate(key_order, start=1):
        ninfo = nuc_dict[k]
        contour = ninfo.get("contour", None)
        if not contour:
            continue

        # contour is [[x, y], ...] in tile-local coords
        contour_arr = np.asarray(contour, dtype=float)
        xs = contour_arr[:, 0]
        ys = contour_arr[:, 1]

        # skimage.draw.polygon expects (row, col) = (y, x)
        rr, cc = draw_polygon(ys, xs, shape=labels.shape)
        labels[rr, cc] = i

    return labels, key_order


def extract_expanded_geometry(labels_expanded, key_order):
    """
    For each label in labels_expanded, extract contour, centroid, and bbox.

    Parameters
    ----------
    labels_expanded : (H, W) int32 ndarray
        Result of expand_labels(labels, distance=...).
    key_order : list of str
        Nucleus keys corresponding to label IDs 1..N.

    Returns
    -------
    geom : dict
        key -> {
            'contour': [[x, y], ...],
            'centroid': [x_center, y_center],
            'bbox': [[ymin, xmin], [ymax, xmax]],
        }
        Only keys with non-empty regions are included.
    """
    geom = {}

    # We can get regionprops directly from labels_expanded
    props = regionprops(labels_expanded)

    # Map label id -> regionprops
    label_to_region = {int(r.label): r for r in props}

    max_label = labels_expanded.max()

    for label_id in range(1, max_label + 1):
        if label_id > len(key_order):
            # e.g. if some labels never got a nucleus; skip
            continue

        key = key_order[label_id - 1]
        region = label_to_region.get(label_id, None)
        if region is None:
            # Label does not exist in expanded image (e.g. removed)
            continue

        # regionprops uses (row, col) = (y, x)
        y_centroid, x_centroid = region.centroid
        minr, minc, maxr, maxc = region.bbox

        # Extract contour using find_contours on binary mask
        mask = (labels_expanded == label_id)
        contours = find_contours(mask.astype(np.uint8), level=0.5)

        if not contours:
            # Fallback: no contour found, skip or at least store bbox centroid
            contour_xy = []
        else:
            # Choose the longest contour
            longest = max(contours, key=lambda c: c.shape[0])
            # longest is array of shape (M, 2) with (row, col) = (y, x) coords
            ys = longest[:, 0]
            xs = longest[:, 1]

            # Convert back to list of [x, y], rounding to nearest pixel
            contour_xy = [
                [float(x), float(y)] for x, y in zip(xs, ys)
            ]

        geom[key] = {
            "contour": contour_xy,
            "centroid": [float(x_centroid), float(y_centroid)],
            "bbox": [[int(minr), int(minc)], [int(maxr), int(maxc)]],
        }

    return geom


def process_one_json(in_path, out_path, distance_px):
    """
    Load one HoVer-Net JSON tile, expand labels, and save updated JSON.
    """
    with open(in_path, "r") as f:
        data = json.load(f)

    nuc_dict = data.get("nuc", {})
    if not nuc_dict:
        # Nothing to do, just copy file over
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"[SKIP] {os.path.basename(in_path)} (no nuclei)")
        return

    labels, key_order = build_label_image_from_nuc_dict(nuc_dict)
    if labels is None or labels.size == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"[SKIP] {os.path.basename(in_path)} (empty label image)")
        return

    # Expand labels without overlap (scikit-image expand_labels)
    labels_expanded = expand_labels(labels, distance=distance_px)

    # Extract new geometry for each nucleus key
    geom = extract_expanded_geometry(labels_expanded, key_order)

    # Build new nuc dict: keep all other fields, update contour / centroid / bbox
    new_nuc_dict = {}
    for k, ninfo in nuc_dict.items():
        ninfo_new = dict(ninfo)  # shallow copy
        if k in geom:
            g = geom[k]
            # Overwrite nuclear geometry with expanded "cell" geometry
            ninfo_new["contour"] = g["contour"]
            ninfo_new["centroid"] = g["centroid"]
            ninfo_new["bbox"] = g["bbox"]
        else:
            # If no expanded region found, keep original (nuclear) geometry
            pass
        new_nuc_dict[k] = ninfo_new

    new_data = dict(data)
    new_data["nuc"] = new_nuc_dict

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(new_data, f)

    print(f"[OK] {os.path.basename(in_path)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Expand HoVer-Net nuclei labels into cell-sized masks "
                    "using skimage.segmentation.expand_labels."
    )
    parser.add_argument(
        "--in-json-dir",
        required=True,
        help="Directory containing input HoVer-Net JSON tiles (yXXXX_xYYYY.json).",
    )
    parser.add_argument(
        "--out-json-dir",
        required=True,
        help="Directory to write expanded JSON tiles (same filenames).",
    )
    parser.add_argument(
        "--distance-px",
        type=float,
        default=8.0,
        help="Expansion distance in pixels passed to expand_labels (default: 8).",
    )

    args = parser.parse_args()

    in_dir = args.in_json_dir
    out_dir = args.out_json_dir
    distance_px = float(args.distance_px)

    json_paths = sorted(glob.glob(os.path.join(in_dir, "y*_x*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No HoVer-Net JSON tiles found in {in_dir}")

    print(f"Found {len(json_paths)} JSON tiles in {in_dir}")
    print(f"Writing expanded JSON tiles to {out_dir}")
    print(f"Using expand_labels distance = {distance_px} px\n")

    for jp in json_paths:
        rel = os.path.relpath(jp, in_dir)
        op = os.path.join(out_dir, rel)
        process_one_json(jp, op, distance_px=distance_px)


if __name__ == "__main__":
    main()
