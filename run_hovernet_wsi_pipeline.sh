#!/bin/bash
# generate_and_plot_cell_masks/run_hovernet_wsi_pipeline.sh

set -euo pipefail

# ==========================================================================================================
# Change this part to reflect your CATCH result path + where you want to output the masks
# ==========================================================================================================
SAMPLE_NAME="Xenium_Prime_Cervical_Cancer_FFPE_he_image"
GPU_ID="0"

# Path to the WSI (H&E) image
INPUT_IMG="/project/CATCH/dataset/10x_cervical_cancer/${SAMPLE_NAME}/he_raw.tif"

# Where to store tiles + HoVer-Net outputs (json/mat/overlays)
BASE_OUT_DIR="/project/CATCH/dataset/for_hovernet/hovernet_results_lm"

# Optional: change model / nr_types / type_info.json for your dataset
MODEL_PATH="./pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar"
NR_TYPES=6
TYPE_INFO_JSON="type_info.json"

# ========================================================
# usually no need to edit this part 
# ========================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TILE_DIR="${BASE_OUT_DIR}/tiles/${SAMPLE_NAME}"
HOVER_OUT_DIR="${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}"

mkdir -p "$HOVER_OUT_DIR"
LOG_FILE="${HOVER_OUT_DIR}/${SAMPLE_NAME}_pipeline.log"

# Derived paths
MAT_DIR="${HOVER_OUT_DIR}/mat"
OVERLAY_TILE_DIR="${HOVER_OUT_DIR}/overlay"
BINARY_MASK_TILE_DIR="${HOVER_OUT_DIR}/binary_masks"
FINAL_MASK_PATH="${HOVER_OUT_DIR}/${SAMPLE_NAME}_mask.png"
FINAL_OVERLAY_PATH="${HOVER_OUT_DIR}/${SAMPLE_NAME}_overlay.png"

# dims.txt (store outside tile dir to avoid conflicts)
DIMS_FILE="${BASE_OUT_DIR}/tiles/${SAMPLE_NAME}_dims.txt"

run_and_log() {
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running: $*" | tee -a "$LOG_FILE"
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

echo "Starting HoVer-Net WSI pipeline for $SAMPLE_NAME at $(date)" > "$LOG_FILE"

# Remove any conflicting dims.txt in tile dir
if [ -f "${TILE_DIR}/dims.txt" ]; then
    echo "Removing conflicting dims.txt from tile directory..." | tee -a "$LOG_FILE"
    rm "${TILE_DIR}/dims.txt"
fi

# --- STEP 1: TILING ---
echo "Step 1: Tiling..." | tee -a "$LOG_FILE"
run_and_log python wsi_utils.py tile \
    --input "$INPUT_IMG" \
    --out_dir "$TILE_DIR" \
    --dims_out "$DIMS_FILE"

# --- STEP 2: HOVERNET INFERENCE (WSI via tiles) ---
echo "Step 2: Running HoVer-Net Inference..." | tee -a "$LOG_FILE"
run_and_log python run_infer.py \
    --gpu="$GPU_ID" \
    --nr_types="$NR_TYPES" \
    --type_info_path="$TYPE_INFO_JSON" \
    --batch_size=64 \
    --model_mode=fast \
    --model_path="$MODEL_PATH" \
    --nr_inference_workers=8 \
    --nr_post_proc_workers=16 \
    tile \
    --input_dir="$TILE_DIR" \
    --output_dir="$HOVER_OUT_DIR" \
    --mem_usage=0.1 \
    --draw_dot \
    --save_qupath

# --- STEP 3: CONVERT MAT -> TILE MASKS ---
echo "Step 3: Converting .mat -> tile masks..." | tee -a "$LOG_FILE"
run_and_log python wsi_utils.py mats \
    --mat_dir "$MAT_DIR" \
    --mask_out_dir "$BINARY_MASK_TILE_DIR"

# --- STEP 4: STITCH FINAL WHOLE-SLIDE MASK + OVERLAY ---
echo "Step 4: Stitching final mask + overlay..." | tee -a "$LOG_FILE"

run_and_log python wsi_utils.py stitch \
    --tile_dir "$BINARY_MASK_TILE_DIR" \
    --output "$FINAL_MASK_PATH" \
    --mode "L" \
    --dims_file "$DIMS_FILE"

run_and_log python wsi_utils.py stitch \
    --tile_dir "$OVERLAY_TILE_DIR" \
    --output "$FINAL_OVERLAY_PATH" \
    --mode "RGB" \
    --dims_file "$DIMS_FILE"

echo "Pipeline complete for $SAMPLE_NAME. Log saved to $LOG_FILE" | tee -a "$LOG_FILE"
