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
    sample = "P24_LUAD_Visium" # name of the sample you ran with CATCH
    he_path = f"/project/CATCH/xiaokang/CATCH_project_main_result/results/fuduanData_multi_55_samples/{sample}/he.tiff" # the directory where the raw H&E image is saved
    hovernet_json_dir = f"/project/CATCH/lexi/hovernet_results/hover_net_out/{sample}/json_expanded" # where the results from the last step (expansion) are

    minimal_dir = f"/project/CATCH/xiaokang/CATCH_project_main_result/results/fuduanData_multi_55_samples/{sample}/clustering_result_old" # the clustering_hier_result.h5ad file from CATCH
    h5ad_filename = "clustering_hier_new_result" # name of the h5ad file (sometimes it's clustering_hier_result, sometimes it's clustering_result)
    minimal_h5ad_path = os.path.join(minimal_dir, f"{h5ad_filename}.h5ad")

    save_dir = f"/project/CATCH/lexi/hovernet_results/plots/{sample}" # where you want to save the plots
    base_cluster_key = "hier_level2_K_2" # which cluster key to plot
    
    downsample_factor = 1.0  # the full resolution plot will be the same size as the original H&E. Downsample as needed so as to not fill up the storage space.

    color_dict = {
        0: [255,127,14], 1: [44,160,44], 2: [214,39,40], 3: [148,103,189],
        4: [140,86,75], 5: [227,119,194], 6: [127,127,127], 7: [188,189,34],
        8: [23,190,207], 9: [174,199,232], 10: [255,187,120], 11: [152,223,138],
        12: [255,152,150], 13: [197,176,213], 14: [196,156,148], 15: [247,182,210], 
        16: [199,199,199], 17: [219,219,141], 18: [158,218,229], 19: [148,0,211], 
        20: [0,128,255], 21: [255,0,128], 22: [255,250,200], 23: [230,190,255], 
        24: [64, 255, 255], 25: [210,245,60], 26: [51,255,51], 27: [255,255,0],
        28: [255,204,0], 29: [255, 228, 181],
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
        out_formats="jpg", # change this if you only want one format
        dpi=200,
    )

    plot_selected_cluster_mask_on_he(
        sample=sample,
        he_path=he_path,
        hovernet_json_dir=hovernet_json_dir,
        save_dir=save_dir,
        minimal_h5ad_path=minimal_h5ad_path,
        cluster_key=base_cluster_key,
        selected_cluster=3,
        mode="masked",
        background_style="he_faded",
        downsample_factor=1,
        out_formats="jpg",
    )


if __name__ == "__main__":
    import os
    main()
