# `cell_masking`

This is an end-to-end mini-pipeline that is built on top of hover_net; the plotting step assumes you have run CATCH and are plotting its clustering results:

1. Runs **HoVer-Net** on a whole-slide image (WSI) to get per-tile nuclei instance JSONs.
2. Expands nuclei contours into approximate **whole-cell masks**.
3. Plots **clustering results** from CATCH on top of cell masks (using your own `.h5ad` clustering output).

---

## 1. Installation

```bash
conda env create -f hover_net/environment.yml
conda activate hovernet

# HoVer-Net requires a specific torch/torchvision combo
pip install torch==1.6.0 torchvision==0.7.0

# Extra packages needed for the cell-masking + plotting pipeline
pip install scanpy pyvips scikit-image matplotlib
```
> If you get `ImportError: no module named 'pyvips'` or similar, make sure system `libvips` is installed (e.g. via `apt`, `yum`, or `brew`).

### 1.1 Pretrained HoVer-Net weights

The file `hover_net/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar` is too large for GitHub (>100 MB), so it is **not** included in this repository.

Please download them from the official HoVer-Net repository: [HoVer-Net GitHub – Model Weights](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view?pli=1) and place it in the following path: hover_net/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar

---

## 2. Directory structure

```text
cell_masking
├── README.md # this file
├── hover_net/ # contains the original hover_net modules. Do not change scripts in this part.
│    ├── run_infer.py
│    ├── wsi_utils.py
│    ├── pretrained/
│    ├── type_info.json
│    ├── ...
└── run/
    ├── run_hovernet_wsi_pipeline.sh      # getting nuclei masks with hover_net
    ├── expand_hovernet_mask.py         # expanding the hover_net nuclei mask to get cell masks
    ├── run_expansion.sh               # example script to call expand_hovernet_cells.py
    ├── plot_clusters_on_cell_masks.py    # plotting library
    └── run_plot_cell_masks_example.py    # example script to call plot_clusters_on_cell_masks.py
```

---

## 3. Script details

- `run_hovernet_wsi_pipeline.sh`  
  Tiling + HoVer-Net WSI inference + mat→mask + stitching.  
  Produces:
  - tiles in `${BASE_OUT_DIR}/tiles/<SAMPLE_NAME>/`
  - HoVer-Net outputs in `${BASE_OUT_DIR}/hover_net_out/<SAMPLE_NAME>/`:
    - `mat/`
    - `overlay/`
    - `json/` (nuclei)
    - `<SAMPLE_NAME>_mask.png` (stitched binary mask)
    - `<SAMPLE_NAME>_overlay.png` (stitched overlay)

- `expand_hovernet_mask.py`  
  Python script that:
  - Reads per-tile `json` from HoVer-Net (nuclei contours).
  - Uses `skimage.segmentation.expand_labels` to grow nuclei into cell-sized regions.
  - Writes updated JSONs with expanded `contour`, `centroid`, `bbox`.

- `run_expansion.sh`  
  Thin wrapper around `expand_hovernet_mask.py`. You only edit:
  - `SAMPLE_NAME`
  - `BASE_OUT_DIR`
  - `EXPANSION_DIST` (pixels)

- `plot_clusters_on_cell_masks.py`  
  Library with the main function:
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
      spatial_scale_factor=16.0,
      max_match_dist_px=None,
      downsample_factor=1.0,
      background_color=(0, 0, 0),
      label_to_color=None,
      legend_font_rel=0.025,
      legend_min_font_px=12,
      out_formats=("png",),
      dpi=200,
  )

---

## 4. How to run on a new dataset

### Step 0 – Required inputs

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


---

### Step 1 – Run HoVer-Net WSI Pipeline

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

From the HoVer-Net repo root, run:

```bash
cd /path/to/hover_net
bash generate_and_plot_cell_masks/run_hovernet_wsi_pipeline.sh
# or: bash cell_masking/run/run_hovernet_wsi_pipeline.sh  (depending on folder naming)
```

#### 1.1. Outputs to check

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

Open:

```text
run/run_expansion.sh
```

At the top, edit:

```bash
SAMPLE_NAME="YourSampleName"
BASE_OUT_DIR="/path/to/hovernet_results"

EXPANSION_DIST=8
```

Then run:

```bash
cd /path/to/hover_net/generate_and_plot_cell_masks/run
bash run_expansion.sh
```

---

### Step 3 – Plot Clusters on Cell Masks

Open:

```text
run/run_plot_cell_masks_example.py
```

Edit (or copy and adapt) an example block. Then run:

```bash
cd /path/to/hover_net/generate_and_plot_cell_masks/run
python run_plot_cell_masks_example.py
# or, if you have a small wrapper script:
# bash run_plot_cell_masks.sh
```

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
