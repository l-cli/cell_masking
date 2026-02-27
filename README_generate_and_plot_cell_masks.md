# `generate_and_plot_cell_masks`

End-to-end mini-pipeline built on top of **HoVer-Net** to:

1. Run HoVer-Net on a whole-slide image (WSI) to get per-tile nuclei instance JSONs.  
2. Expand nuclei contours into approximate **whole-cell masks**.  
3. Plot **clustering results** (e.g. from CATCH) on top of the cell masks using your `.h5ad` or `(coords, labels)`.

This folder is meant to live **inside** the HoVer-Net repository, reusing:

- `run_infer.py`
- `wsi_utils.py`
- `pretrained/` (HoVer-Net weights)
- `type_info.json` (nuclear types and colors)

---

## 1. Installation

### 1.1. Create the HoVer-Net conda environment

```bash
conda env create -f hover_net/environment.yml
conda activate hovernet

# HoVer-Net requires this torch/torchvision combo
pip install torch==1.6.0 torchvision==0.7.0
```

### 1.2. Extra dependencies for cell masking + plotting

The `environment.yml` includes:

- `python=3.6.12`
- `openslide`
- and installs `-r requirements.txt`
- plus `openslide-python==1.1.2` (via pip)

For the **cell-masking and plotting** pipeline, you also need:

- `scanpy` – reading `.h5ad` clustering outputs  
- `pyvips` – large WSI I/O  
- `scikit-image` – label expansion (`expand_labels`, `regionprops`, `find_contours`)  
- `matplotlib` – figure export  

Install these (if they’re not already in `requirements.txt`):

```bash
conda activate hovernet
pip install scanpy pyvips scikit-image matplotlib
```

> If you get `ImportError: no module named 'pyvips'` or similar, make sure system `libvips` is installed (e.g. via `apt`, `yum`, or `brew`).

---

## 2. Directory Structure

Example layout:

```text
cell_masking/
├── README.md                 # this file
├── hover_net/                # original hover_net repo (do not edit its core scripts)
│   ├── run_infer.py
│   ├── wsi_utils.py
│   ├── pretrained/
│   ├── type_info.json
│   └── ...
└── run/                      # scripts specific to the cell mask + plotting pipeline
    ├── run_hovernet_wsi_pipeline.sh   # run HoVer-Net on WSI to get nuclei masks
    ├── expand_hovernet_mask.py        # expand nuclei labels to approximate cell masks
    ├── run_expansion.sh               # example wrapper for expand_hovernet_mask.py
    ├── plot_clusters_on_cell_masks.py # plotting library
    └── run_plot_cell_masks_example.py # example caller for plotting
```

(If you prefer, rename `cell_masking/` to `generate_and_plot_cell_masks/`; the structure is the same.)

---

## 3. Scripts Overview

### 3.1. `run_hovernet_wsi_pipeline.sh`

- Runs the **full HoVer-Net WSI pipeline**:
  1. Tile the WSI (H&E image).
  2. Run HoVer-Net inference on tiles.
  3. Convert `.mat` files to tile-level binary masks.
  4. Stitch tiles into whole-slide mask and overlay.

- Outputs:
  - Tiles: `${BASE_OUT_DIR}/tiles/<SAMPLE_NAME>/`
  - HoVer-Net results: `${BASE_OUT_DIR}/hover_net_out/<SAMPLE_NAME>/`
    - `mat/`
    - `overlay/`
    - `json/` (per-tile nuclei)
    - `<SAMPLE_NAME>_mask.png` (stitched mask)
    - `<SAMPLE_NAME>_overlay.png` (stitched overlay)

### 3.2. `expand_hovernet_mask.py`

- Reads **HoVer-Net JSON tiles**:

  ```json
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
      ...
    }
  }
  ```

- Builds a label image, runs:

  ```python
  from skimage.segmentation import expand_labels
  labels_expanded = expand_labels(labels, distance=distance_px)
  ```

- For each label, extracts new `contour`, `centroid`, and `bbox`, and writes updated JSONs with the **same structure** but cell-sized regions.

### 3.3. `run_expansion.sh`

- Simple wrapper around `expand_hovernet_mask.py`.
- You edit:
  ```bash
  SAMPLE_NAME="..."
  BASE_OUT_DIR="..."
  EXPANSION_DIST=8
  ```
- It reads from:
  ```bash
  ${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/json
  ```
  and writes to:
  ```bash
  ${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/json_expanded
  ```

### 3.4. `plot_clusters_on_cell_masks.py`

- Library that exposes:

  ```python
  plot_clusters_on_cell_masks(
      sample,
      he_path,
      hovernet_json_dir,
      save_dir,
      minimal_h5ad_path=None,
      coords=None,
      labels=None,
      cluster_key="hier_kmeans",
      vis_basis="spatial",
      spatial_scale_factor=16.0,   # e.g. Visium spot coords -> H&E pixels
      max_match_dist_px=None,      # skip nuclei farther than this from any spot
      downsample_factor=1.0,       # final image only (0 < d <= 1)
      background_color=(0, 0, 0),
      label_to_color=None,         # {cluster_label -> "#RRGGBB" or (R,G,B)}
      legend_font_rel=0.025,
      legend_min_font_px=12,
      out_formats=("png",),
      dpi=200,
  )
  ```

- Takes either:
  - A `minimal_h5ad_path` with `ad.obsm[vis_basis]` and `ad.obs[cluster_key]`, **or**
  - Explicit `coords` and `labels` arrays.

- Assigns each nucleus/cell the **nearest cluster label** and draws the cell polygons colored by cluster on a black background, with a legend on the right.

### 3.5. `run_plot_cell_masks_example.py`

- Example driver script that:
  - Sets up paths for:
    - `sample`
    - `he_path`
    - `hovernet_json_dir` (usually `json_expanded`)
    - `minimal_h5ad_path`
    - `save_dir`
    - `spatial_scale_factor`
    - `max_match_dist_px`
    - `cluster_key`
    - `label_to_color` (color map)
  - Calls `plot_clusters_on_cell_masks(...)` for one or more samples.

- Can be wrapped by a tiny bash script (e.g. `run_plot_cell_masks.sh`) that sets env variables:

  ```bash
  export COLOR_MODE="grey"
  export DOWNSAMPLE_FACTOR="0.5"
  python run_plot_cell_masks_example.py
  ```

---

## 4. How to Run on a New Dataset

This section is structured step-by-step with clearly separated **inputs**, **files to edit**, and **outputs**.

---

### Step 0 – Prepare Data and Clustering

#### 0.1. Required inputs

- **WSI (H&E image)**  
  - Format readable by `pyvips` (e.g. `.tif`, `.tiff`, `.svs` converted, etc.).
  - Path example:
    ```text
    /project/CATCH/dataset/<dataset_name>/<sample_name>/he_raw.tif
    ```

- **Clustering result**  
  Either:
  1. `.h5ad` with:
     - `ad.obsm[vis_basis]` (e.g. `"spatial"`), shape `(N, 2)` with `(x, y)` coordinates.
     - `ad.obs[cluster_key]` containing your cluster labels (strings or integers).

  **or**
  2. Two arrays in Python:
     - `coords` – shape `(N, 2)`, `(x, y)` in spot/cluster coordinate space.
     - `labels` – length `N`, cluster labels.

#### 0.2. Coordinate scaling

- You must know how to map clustering coordinates to H&E pixels:
  - `spatial_scale_factor` is used to convert from `vis_basis` units to H&E pixel coordinates.
  - For Visium, a typical value is `16.0` (adjust if needed).

---

### Step 1 – Run HoVer-Net WSI Pipeline

#### 1.1. Files to edit

Open:

```text
run/run_hovernet_wsi_pipeline.sh
```

At the top, edit:

```bash
SAMPLE_NAME="YourSampleName"
GPU_ID="0"   # or another GPU id

# Path to the WSI
INPUT_IMG="/path/to/your_dataset/${SAMPLE_NAME}/he_raw.tif"

# Where tiles & HoVer-Net outputs will be stored
BASE_OUT_DIR="/path/to/hovernet_results"

# Model and type settings
MODEL_PATH="./pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar"
NR_TYPES=6
TYPE_INFO_JSON="type_info.json"
```

#### 1.2. Run the script

From the HoVer-Net repo root:

```bash
cd /path/to/hover_net
bash generate_and_plot_cell_masks/run_hovernet_wsi_pipeline.sh
# or: bash cell_masking/run/run_hovernet_wsi_pipeline.sh  (depending on folder naming)
```

#### 1.3. Outputs to check

- **Tiles**:

  ```text
  ${BASE_OUT_DIR}/tiles/${SAMPLE_NAME}/
      y00000000_x00000000.png
      y00002048_x00000000.png
      ...
  ```

- **HoVer-Net outputs**:

  ```text
  ${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/
      mat/                  # .mat instance maps
      overlay/              # per-tile overlays
      json/                 # per-tile nuclei JSONs
      ${SAMPLE_NAME}_mask.png     # stitched whole-slide mask
      ${SAMPLE_NAME}_overlay.png  # stitched whole-slide overlay
  ```

If these look good, proceed to expansion.

---

### Step 2 – Expand Nuclei to Cell Masks

#### 2.1. Files to edit

Open:

```text
run/run_expansion.sh
```

At the top, edit:

```bash
SAMPLE_NAME="YourSampleName"
BASE_OUT_DIR="/path/to/hovernet_results"

# Expansion distance in pixels (typical: 6–10)
EXPANSION_DIST=8
```

#### 2.2. Run the expansion

```bash
cd /path/to/hover_net/generate_and_plot_cell_masks/run
bash run_expansion.sh
```

#### 2.3. Inputs and outputs

- **Input JSON directory**:

  ```bash
  IN_JSON_DIR=${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/json
  ```

- **Output JSON directory**:

  ```bash
  OUT_JSON_DIR=${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/json_expanded
  ```

Each JSON in `json_expanded`:

- Uses the same `"nuc"` key structure as original HoVer-Net outputs.
- But `"contour"`, `"centroid"`, and `"bbox"` correspond to the **expanded** cell-sized region.

> If you change `EXPANSION_DIST`, you can simply re-run `run_expansion.sh`.  
> It only depends on the original `json` directory and does not modify it.

---

### Step 3 – Plot Clusters on Cell Masks

#### 3.1. Files to edit

Open:

```text
run/run_plot_cell_masks_example.py
```

Edit (or copy and adapt) an example block:

```python
sample = "YourSampleName"

he_path = "/path/to/your_dataset/YourSampleName/he_raw.tif"

hovernet_json_dir = (
    f"/path/to/hovernet_results/hover_net_out/{sample}/json_expanded"
)

minimal_dir = (
    "/path/to/your_clustering_intermediates/hier/merged/"
    f"{sample}"
)
h5ad_filename = "your_minimal_h5ad_basename"
minimal_h5ad_path = os.path.join(minimal_dir, f"{h5ad_filename}.h5ad")

save_dir = "/path/to/output/plots/clusters_on_cell_masks/YourExperimentName"

spatial_scale_factor = 16.0
max_match_dist_px = 16.0
base_cluster_key = "your_cluster_column_in_obs"

colormap_custom = {
    "cluster_1": "#6db5f2",
    "cluster_2": "#f8f968",
    "cluster_3": "#a3a3a3",
    "unknown":   "#a3a3a3",
}

plot_clusters_on_cell_masks(
    sample=sample,
    he_path=he_path,
    hovernet_json_dir=hovernet_json_dir,
    save_dir=save_dir,
    minimal_h5ad_path=minimal_h5ad_path,
    cluster_key=base_cluster_key,
    vis_basis="spatial",
    spatial_scale_factor=spatial_scale_factor,
    max_match_dist_px=max_match_dist_px,
    downsample_factor=downsample_factor,
    background_color=(0, 0, 0),
    label_to_color=colormap_custom,
    legend_font_rel=0.025,
    legend_min_font_px=12,
    out_formats=("png", "pdf"),
    dpi=200,
)
```

#### 3.2. Optional environment variables

At the shell:

```bash
export COLOR_MODE="grey"         # only used if your script switches on COLOR_MODE
export DOWNSAMPLE_FACTOR="0.5"   # 1.0 = full-res; 0.5 = half; 0.25 = quarter
```

#### 3.3. Run the plotting script

```bash
cd /path/to/hover_net/generate_and_plot_cell_masks/run
python run_plot_cell_masks_example.py
# or, if you have a small wrapper script:
# bash run_plot_cell_masks.sh
```

#### 3.4. Outputs

The script will create:

```text
save_dir/<sample>/<cluster_key>/clusters_on_cell_masks.png
save_dir/<sample>/<cluster_key>/clusters_on_cell_masks.pdf
# (plus any other formats you listed in out_formats)
```

The output images:

- Show each **cell** (expanded HoVer-Net region) filled with the color corresponding to its assigned cluster label.
- Include a legend strip on the right, listing label → color mapping.

---

## 5. Summary of Files to Change for a New Dataset

- **HoVer-Net WSI processing**

  - File: `run_hovernet_wsi_pipeline.sh`
  - Change:
    - `SAMPLE_NAME`
    - `INPUT_IMG`
    - `BASE_OUT_DIR`
    - (Optionally) `GPU_ID`, `MODEL_PATH`, `NR_TYPES`, `TYPE_INFO_JSON`

- **Nucleus → Cell expansion**

  - File: `run_expansion.sh`
  - Change:
    - `SAMPLE_NAME`
    - `BASE_OUT_DIR`
    - `EXPANSION_DIST`

- **Cluster overlay plotting**

  - File: `run_plot_cell_masks_example.py`
  - Change:
    - `sample`
    - `he_path`
    - `hovernet_json_dir` (usually `json_expanded`)
    - `minimal_h5ad_path` (or `coords`/`labels`)
    - `save_dir`
    - `spatial_scale_factor`
    - `max_match_dist_px`
    - `cluster_key`
    - `label_to_color` (cluster→color mapping)

Once these three pieces are configured, running:

1. `run_hovernet_wsi_pipeline.sh`
2. `run_expansion.sh`
3. `run_plot_cell_masks_example.py`

will take you from a raw WSI + clustering result to a final **cell-level cluster overlay** figure.
