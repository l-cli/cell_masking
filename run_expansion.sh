#!/bin/bash
# generate_and_plot_cell_masks/run_expand_cells.sh

set -euo pipefail

# ============================
# edit parameters in this part
# ============================
SAMPLE_NAME="Xenium_Prime_Cervical_Cancer_FFPE_he_image"
BASE_OUT_DIR="/project/CATCH/dataset/for_hovernet/hovernet_results_lm"
EXPANSION_DIST=6  # pixels, 4-6 is typical, but you can adjust based on how big you expect the cells to be

IN_JSON_DIR="${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/json" # change if your hovernet results are stored elsewhere
OUT_JSON_DIR="${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}/json_expanded" # change if you want to save the expanded mask elsewhere

# ============================
# running the script
# ============================
echo "Expanding HoVer-Net nuclei for sample ${SAMPLE_NAME}"
echo "Input JSON dir : ${IN_JSON_DIR}"
echo "Output JSON dir: ${OUT_JSON_DIR}"
echo "Distance (px)  : ${EXPANSION_DIST}"

python expand_hovernet_cells.py \
  --in-json-dir  "$IN_JSON_DIR" \
  --out-json-dir "$OUT_JSON_DIR" \
  --distance-px  "$EXPANSION_DIST"
