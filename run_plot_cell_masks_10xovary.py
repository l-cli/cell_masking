#!/usr/bin/env python
"""
Script to plot clusters on expanded HoVer-Net cell masks for 10x Xenium_Prime_Human_Ovary_FF

"""

import os
from plot_clusters_on_cell_masks import plot_clusters_on_cell_masks, plot_selected_cluster_mask_on_he


def main():
    # ==========================================================
    # Example: 10x Cervical cancer
    # ==========================================================
    sample = "Xenium_Prime_Human_Ovary_FF_he_image" # name of the sample you ran with CATCH
    he_path = f"/project/CATCH3/lexi/datasets/{sample}/he.tif" # the directory where the raw H&E image is saved
    hovernet_json_dir = f"/project/CATCH/dataset/for_hovernet/hovernet_results_lm/hover_net_out/{sample}/json_expanded" # where the results from the last step (expansion) are
    #hovernet_json_dir = f"/project/CATCH/dataset/for_hovernet/hovernet_results_lm/hover_net_out/{sample}/json" # where the results from the last step (expansion) are

    minimal_dir = f"/project/CATCH/lexi/intermediates/10x_ovary_single_1_sample/hier/merged/Xenium_Prime_Human_Ovary_FF_he_image/" # the clustering_hier_result.h5ad file from CATCH
    h5ad_filename = "minimal_hier_kmeans_lvl2_merged" # name of the h5ad file (sometimes it's clustering_hier_result, sometimes it's clustering_result)
    minimal_h5ad_path = os.path.join(minimal_dir, f"{h5ad_filename}.h5ad")

    save_dir = f"/project/CATCH/lexi/hovernet_results/plots/{sample}" # where you want to save the plots
    base_cluster_key = "hier_kmeans_lvl2_merged" # which cluster key to plot
    
    downsample_factor = 0.5  # the full resolution plot will be the same size as the original H&E. Downsample as needed so as to not fill up the storage space.

    color_dict = {
        "cancer": "#6db5f2", 
        "s2 CAF": "#a3a3a3", 
        "s1 CAF": "#f8f968", 
        "s1 - needs confirmation": "#a3a3a3", 
        "s2 - needs confirmation": "#a3a3a3",
        "unknown": "#a3a3a3",
    }
    background_color = (0, 0, 0) # change this if you want to plot over a differently colored background, default is black

    print("Running cluster overlay for sample:", sample)

    plot_clusters_on_cell_masks(
        sample=sample,
        he_path=he_path,
        hovernet_json_dir=hovernet_json_dir,
        save_dir=save_dir,
        minimal_h5ad_path=minimal_h5ad_path,
        cluster_key=base_cluster_key,
        vis_basis="spatial",
        spatial_scale_factor=16.0,
        max_match_dist_px=16.0, # since we match the nuclei centroid with the closest cluster label, nuclei farther than this from any cluster coord are skipped
        downsample_factor=downsample_factor,
        background_color=background_color,
        label_to_color=color_dict, 
        legend_font_rel=0.025,
        legend_min_font_px=12,
        dpi=200,
    )

    # plot_selected_cluster_mask_on_he(
    #     sample=sample,
    #     he_path=he_path,
    #     hovernet_json_dir=hovernet_json_dir,
    #     save_dir=save_dir,
    #     minimal_h5ad_path=minimal_h5ad_path,
    #     cluster_key=base_cluster_key,
    #     selected_cluster=5,
    #     mode="masked",
    #     background_style="he_faded",
    #     downsample_factor=1,
    # )


if __name__ == "__main__":
    import os
    main()
