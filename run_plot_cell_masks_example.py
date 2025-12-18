#!/usr/bin/env python
"""
Example driver script to plot clusters on expanded HoVer-Net cell masks.

Edit the paths and mappings in `main()` for your dataset, then run.
"""

import os
from plot_clusters_on_cell_masks import plot_clusters_on_cell_masks


def main():
    # ==========================================================
    # Example: 10x Cervical cancer
    # ==========================================================
    sample = "Xenium_Prime_Cervical_Cancer_FFPE_he_image" # name of the sample you ran with CATCH
    he_path = "../10x_cervical_cancer/Xenium_Prime_Cervical_Cancer_FFPE_he_image/he_raw.tif" # the directory where the raw H&E image is saved
    hovernet_json_dir = f"../hovernet_results_lm/hover_net_out/{sample}/json_expanded" # where the results from the last step (expansion) are

    minimal_dir = f"10x_cervical_cancer_single_1_sample/{sample}" # the clustering_hier_result.h5ad file from CATCH
    h5ad_filename = "minimal_hier_kmeans_lvl2_merged" # name of the h5ad file (sometimes it's clustering_hier_result, sometimes it's clustering_result)
    minimal_h5ad_path = os.path.join(minimal_dir, f"{h5ad_filename}.h5ad")

    save_dir = "../clusters_on_cell_masks/" # where you want to save the plots
    base_cluster_key = "hier_kmeans_lvl2_merged" # which cluster key to plot
    
    downsample_factor = 1.0  # the full resolution plot will be the same size as the original H&E. Downsample as needed so as to not fill up the storage space.

    colormap_cervical = { # what colors to plot for each cluster/annotation label, if none, will be automatically generated with a color palette
        "cancer": "#6db5f2",
        "s1 CAF": "#f8f968",
        "endothelium": "#a3a3a3",
        "epithelium": "#a3a3a3",
        "muscle": "#a3a3a3",
        "non-CAF fibroblasts": "#a3a3a3",
        "s2 + s4 CAF": "#a3a3a3",
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
        label_to_color=colormap_cervical, 
        legend_font_rel=0.025,
        legend_min_font_px=12,
        out_formats=("png", "pdf"), # change this if you only want one format
        dpi=200,
    )


if __name__ == "__main__":
    import os
    main()
