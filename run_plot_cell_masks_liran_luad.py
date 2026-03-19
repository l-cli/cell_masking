#!/usr/bin/env python
"""
Script to plot numerical values on expanded HoVer-Net cell masks for LUAD for Meowcat
"""

import os
import numpy as np
import anndata as ad
from plot_clusters_on_cell_masks import plot_clusters_on_cell_masks, plot_numerical_values_on_cell_masks


def main():
    sample = "luad_shared_with_lexi"
    he_path = f"/project/CATCH3/liran/{sample}/he.tiff"
    hovernet_json_dir = f"/project/CATCH/lexi/hovernet_results/hover_net_out/{sample}/json"

    minimal_dir = f"/project/CATCH3/liran/{sample}"
    h5ad_filename = "predicted_3phase_Xenium_adata_cellbin_analysis_qv20"
    minimal_h5ad_path = os.path.join(minimal_dir, f"{h5ad_filename}.h5ad")

    save_dir = f"/project/CATCH/lexi/hovernet_results/plots/{sample}"
    
    downsample_factor = 0.5
    spatial_xy_order = "xy"

    # plot both black and white backgrounds
    background_colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0)
    }
    
    for background_name, background_color in background_colors.items():

        save_dir_with_bg = os.path.join(save_dir, f'{background_name}_bg')
        os.makedirs(save_dir_with_bg, exist_ok=True)

        ##########################
        # 1. Categorical labels
        ##########################
        base_cluster_key = "cdan_label"
        color_dict = {
            "NonTumor_Epi": "#ff0000",
            "Tumor_Epi": "#f97a00",
            "Plasma": "#00b8a9",
            "T": "#a8df8e",
            "NK": "#ffef5f",
            "Myeloid": "#000080",
            "B": "#f875aa",
            "Stromal": "#a7aae1"
        }

        print("Running cluster overlay for sample:", sample)

        plot_clusters_on_cell_masks(
            sample=sample,
            he_path=he_path,
            hovernet_json_dir=hovernet_json_dir,
            save_dir=save_dir_with_bg,
            minimal_h5ad_path=minimal_h5ad_path,
            cluster_key=base_cluster_key,
            vis_basis="spatial",
            spatial_scale_factor=16.0,
            spatial_xy_order="xy",
            max_match_dist_px=16.0,
            downsample_factor=downsample_factor,
            background_color=background_color,
            label_to_color=color_dict,
            legend_font_rel=0.025,
            legend_min_font_px=12,
            plot_title=base_cluster_key,
            dpi=200,
        )

        # ==========================================================
        # 2. Numerical column
        # ==========================================================
        value_key = "cdan_probs"

        # Set explicitly:
        #   "numerical"   -> force numerical plotting
        #   "categorical" -> force categorical plotting
        #   None          -> fallback inference
        column_kind = "numerical"

        cmap_name = "viridis"
        vmin = 0.0
        vmax = 1.0

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

        value_keys = {
            0: "NonTumor_Epi",
            1: "Tumor_Epi",
            2: "Plasma",
            3: "T",
            4: "NK",
            5: "Myeloid",
            6: "B",
            7: "Stromal"
        }
        
        # check how many columns in the value_key array in the .h5ad
        adata = ad.read_h5ad(minimal_h5ad_path)
        if value_key in adata.obsm:
            num_cols = adata.obsm[value_key].shape[1]
            print(f"Found {num_cols} columns in obsm['{value_key}'].")
        else:
            raise ValueError(f"Value key '{value_key}' not found in adata.obsm.")
        
        for value_index in range(num_cols):
            print(f"Plotting column {value_index} of obsm['{value_key}']...")

            colorbar_label = value_keys.get(value_index)

            plot_numerical_values_on_cell_masks(
                sample=sample,
                he_path=he_path,
                hovernet_json_dir=hovernet_json_dir,
                save_dir=save_dir_with_bg,
                value_key=value_key,
                minimal_h5ad_path=minimal_h5ad_path,
                value_index=value_index,
                column_kind=column_kind,
                vis_basis="spatial",
                spatial_scale_factor=16.0,
                spatial_xy_order="xy",
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