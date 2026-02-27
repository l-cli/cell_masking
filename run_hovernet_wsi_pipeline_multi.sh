#!/bin/bash
# generate_and_plot_cell_masks/run_hovernet_wsi_pipeline_all.sh
set -euo pipefail

GPU_ID="0"

# Root folder containing SAMPLE_NAME/ (each sample is a subdir)
DATASET_ROOT="/project/CATCH/dataset/melanoma_13"

# --- INPUT: WSI filename can vary ---
# Use a glob pattern to match the WSI inside each sample folder.
# Examples:
#   WSI_GLOB="he_raw.tif"
#   WSI_GLOB="*.tif"
#   WSI_GLOB="*.ndpi"
#   WSI_GLOB="he*.tif"
WSI_GLOB="he.ndpi"

# Optional extra filter (regex applied to basename). Leave empty to disable.
# Example: WSI_REGEX='^(he|HE).*\.tif$'
WSI_REGEX=""

# Where to store tiles + HoVer-Net outputs
BASE_OUT_DIR="/project/CATCH/dataset/for_hovernet/hovernet_results_lm"

MODEL_PATH="./pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar"
NR_TYPES=6
TYPE_INFO_JSON="type_info.json"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run_and_log() {
  local LOG_FILE="$1"; shift
  echo "------------------------------------------------" | tee -a "$LOG_FILE"
  echo "Running: $*" | tee -a "$LOG_FILE"
  echo "------------------------------------------------" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

# Find candidate WSIs exactly one level under DATASET_ROOT:
#   DATASET_ROOT/SAMPLE_NAME/<WSI>
# If your images are deeper, increase -maxdepth.
mapfile -t CANDIDATES < <(find "$DATASET_ROOT" -mindepth 2 -maxdepth 2 -type f -name "$WSI_GLOB" | sort)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "No WSI files matched -name '$WSI_GLOB' under: $DATASET_ROOT" >&2
  exit 1
fi

# If you want to enforce exactly ONE WSI per sample, we’ll pick the first match per sample
# (sorted), and warn if a sample has multiple matches.
declare -A SAMPLE_TO_WSI
declare -A SAMPLE_MULTI_COUNT

for f in "${CANDIDATES[@]}"; do
  base="$(basename "$f")"

  if [ -n "$WSI_REGEX" ]; then
    if ! [[ "$base" =~ $WSI_REGEX ]]; then
      continue
    fi
  fi

  sample_dir="$(dirname "$f")"
  sample_name="$(basename "$sample_dir")"

  if [ -z "${SAMPLE_TO_WSI[$sample_name]+x}" ]; then
    SAMPLE_TO_WSI["$sample_name"]="$f"
    SAMPLE_MULTI_COUNT["$sample_name"]=1
  else
    SAMPLE_MULTI_COUNT["$sample_name"]=$(( SAMPLE_MULTI_COUNT["$sample_name"] + 1 ))
    # keep the first one (already set)
  fi
done

if [ "${#SAMPLE_TO_WSI[@]}" -eq 0 ]; then
  echo "After applying WSI_REGEX, no WSIs remain. Regex was: $WSI_REGEX" >&2
  exit 1
fi

# Iterate samples in name order
mapfile -t SAMPLES_SORTED < <(printf "%s\n" "${!SAMPLE_TO_WSI[@]}" | sort)

for SAMPLE_NAME in "${SAMPLES_SORTED[@]}"; do
  INPUT_IMG="${SAMPLE_TO_WSI[$SAMPLE_NAME]}"

  if [ "${SAMPLE_MULTI_COUNT[$SAMPLE_NAME]}" -gt 1 ]; then
    echo "[WARN] Sample '$SAMPLE_NAME' has ${SAMPLE_MULTI_COUNT[$SAMPLE_NAME]} files matching '$WSI_GLOB'. Using: $INPUT_IMG" >&2
  fi

  echo "================================================================================"
  echo "Processing sample: $SAMPLE_NAME"
  echo "Input image: $INPUT_IMG"
  echo "================================================================================"

  TILE_DIR="${BASE_OUT_DIR}/tiles/${SAMPLE_NAME}"
  HOVER_OUT_DIR="${BASE_OUT_DIR}/hover_net_out/${SAMPLE_NAME}"

  mkdir -p "$HOVER_OUT_DIR"
  LOG_FILE="${HOVER_OUT_DIR}/${SAMPLE_NAME}_pipeline.log"

  MAT_DIR="${HOVER_OUT_DIR}/mat"
  OVERLAY_TILE_DIR="${HOVER_OUT_DIR}/overlay"
  BINARY_MASK_TILE_DIR="${HOVER_OUT_DIR}/binary_masks"
  FINAL_MASK_PATH="${HOVER_OUT_DIR}/${SAMPLE_NAME}_mask.png"
  FINAL_OVERLAY_PATH="${HOVER_OUT_DIR}/${SAMPLE_NAME}_overlay.png"

  # dims file outside tile dir to avoid conflicts
  DIMS_FILE="${BASE_OUT_DIR}/tiles/${SAMPLE_NAME}_dims.txt"

  echo "Starting HoVer-Net WSI pipeline for $SAMPLE_NAME at $(date)" > "$LOG_FILE"

  # Remove any conflicting dims.txt in tile dir
  if [ -f "${TILE_DIR}/dims.txt" ]; then
    echo "Removing conflicting dims.txt from tile directory..." | tee -a "$LOG_FILE"
    rm "${TILE_DIR}/dims.txt"
  fi

  echo "Step 1: Tiling..." | tee -a "$LOG_FILE"
  run_and_log "$LOG_FILE" python wsi_utils.py tile \
    --input "$INPUT_IMG" \
    --out_dir "$TILE_DIR" \
    --dims_out "$DIMS_FILE"

  echo "Step 2: Running HoVer-Net Inference..." | tee -a "$LOG_FILE"
  run_and_log "$LOG_FILE" python run_infer.py \
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

  echo "Step 3: Converting .mat -> tile masks..." | tee -a "$LOG_FILE"
  run_and_log "$LOG_FILE" python wsi_utils.py mats \
    --mat_dir "$MAT_DIR" \
    --mask_out_dir "$BINARY_MASK_TILE_DIR"

  echo "Step 4: Stitching final mask + overlay..." | tee -a "$LOG_FILE"
  run_and_log "$LOG_FILE" python wsi_utils.py stitch \
    --tile_dir "$BINARY_MASK_TILE_DIR" \
    --output "$FINAL_MASK_PATH" \
    --mode "L" \
    --dims_file "$DIMS_FILE"

  run_and_log "$LOG_FILE" python wsi_utils.py stitch \
    --tile_dir "$OVERLAY_TILE_DIR" \
    --output "$FINAL_OVERLAY_PATH" \
    --mode "RGB" \
    --dims_file "$DIMS_FILE"

  echo "Pipeline complete for $SAMPLE_NAME. Log saved to $LOG_FILE" | tee -a "$LOG_FILE"
done