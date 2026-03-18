#!/usr/bin/env python
"""
Script to plot numerical values on expanded HoVer-Net cell masks for LUAD for Meowcat
"""

import os
from plot_clusters_on_cell_masks import plot_numerical_values_on_cell_masks


def main():
    sample = "luad_shared_with_lexi"
    he_path = f"/project/CATCH3/liran/{sample}/he.tiff"
    hovernet_json_dir = f"/project/CATCH/lexi/hovernet_results/hover_net_out/{sample}/json"

    minimal_dir = f"/project/CATCH3/liran/{sample}"
    h5ad_filename = "predicted_3phase_Xenium_adata_cellbin_analysis_qv20"
    minimal_h5ad_path = os.path.join(minimal_dir, f"{h5ad_filename}.h5ad")

    save_dir = f"/project/CATCH/lexi/hovernet_results/plots/"

    # ==========================================================
    # User-set parameters
    # ==========================================================
    value_key = "cdan_probs"

    # Set explicitly:
    #   "numerical"   -> force numerical plotting
    #   "categorical" -> force categorical plotting
    #   None          -> fallback inference
    column_kind = "numerical"

    # For 2D numerical arrays in obsm, choose which column to plot.
    # For cdan_probs with 8 columns, examples might be:
    #   0 NonTumor_Epi
    #   1 Tumor_Epi
    #   2 Plasma
    #   3 T
    #   4 NK
    #   5 Myeloid
    #   6 B
    #   7 Stromal
    value_index = 1

    downsample_factor = 0.25
    background_color = (0, 0, 0)

    cmap_name = "viridis"
    vmin = 0.0
    vmax = 1.0
    colorbar_label = "Tumor_Epi probability"

    print("Running numerical overlay for sample:", sample)

    plot_numerical_values_on_cell_masks(
        sample=sample,
        he_path=he_path,
        hovernet_json_dir=hovernet_json_dir,
        save_dir=save_dir,
        value_key=value_key,
        minimal_h5ad_path=minimal_h5ad_path,
        value_index=value_index,
        column_kind=column_kind,   # user sets this in the script
        vis_basis="spatial",
        spatial_scale_factor=16.0,
        max_match_dist_px=16.0,
        downsample_factor=downsample_factor,
        background_color=background_color,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        save_colorbar=True,
        out_formats=("png",),
        dpi=200,
    )


if __name__ == "__main__":
    main()