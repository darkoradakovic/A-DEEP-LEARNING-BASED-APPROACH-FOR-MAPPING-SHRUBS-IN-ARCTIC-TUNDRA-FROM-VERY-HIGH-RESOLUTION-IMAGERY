

## Load in entire saved_sheet with excel files (see below ### Multimodel plot from multiple files) and plot scatterplots


import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker




savedir_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/scatterplots_models_cover_means'

# Load the data
file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/scatterplots_models_cover_means/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p70_070225.csv'


data = pd.read_csv(file_path)

# modelname = "VGG19"
modelname = file_path.split('/')[-1].split('_')[5]
print(modelname)
multisitename = 'ms10'

# Filter the data for File "ms16"
ms16_data = data[data["File"] == multisitename]
method_name = file_path.split('/')[-1].split('_')[-2]

savename = multisitename+'_'+modelname+'_'+method_name+"_scatter_plots.jpg"
print(savename)
savedir = os.path.join(savedir_path, savename)

# Get unique areas for plotting subplots
ms16_areas = ms16_data["Area"].unique()

# Set up the figure and axis for subplots with reduced size
fig, axes = plt.subplots(len(ms16_areas), 1, figsize=(8, 4 * len(ms16_areas)), sharex=True)

if len(ms16_areas) == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

# Plotting for each area
for i, area in enumerate(ms16_areas):
    area_data = ms16_data[ms16_data["Area"] == area]
    ax = axes[i]

    # Determine y-axis limit
    max_y_value = max(area_data["cover_frac_binary"].max(), area_data["cover_frac_cont"].max(), 0.4)

    # Plot cover_frac_binary with error bars
    ax.errorbar(area_data["Year"], area_data["cover_frac_binary"],
                yerr=[area_data["cover_frac_binary"] - area_data["ci_low_b"],
                      area_data["ci_up_b"] - area_data["cover_frac_binary"]],
                fmt='o', color='red', label="cover_frac_binary", alpha=0.7)

    # Plot cover_frac_cont with error bars
    ax.errorbar(area_data["Year"], area_data["cover_frac_cont"],
                yerr=[area_data["cover_frac_cont"] - area_data["ci_low_cont"],
                      area_data["ci_up_cont"] - area_data["cover_frac_cont"]],
                fmt='o', color='blue', label="cover_frac_cont", alpha=0.7)

    # Add area name in the upper left corner
    ax.text(0.01, 0.95, f"Area: {area}", transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round", alpha=0.3))

    # ax.set_title(f"File: {multisitename}, Area: {area}")
    ax.set_ylim(0, max_y_value + 0.05)
    ax.set_ylabel("Cover Fraction")
    ax.grid(True)
    ax.legend()

# Set common x-axis label and overall title
plt.xlabel("Year")
fig.suptitle("Cover Fraction Trends for File: ms16", fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as a PNG file
# plt.savefig(savedir, dpi=300)  # Save with high resolution (300 DPI)
# plt.savefig(savedir)  # Save with high resolution (300 DPI)

plt.show(block=True)





# all_statistics_df.to_clipboard(index=False)  # Copies DataFrame without the index




### Multimodel plot from multiple files


# File paths (list of your CSV files)
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p70_070225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p70_070225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p70_070225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p70_070225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p70_070225.xlsx'
]

# blocksize 200 filtered
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_100225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_100225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_100225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_100225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_100225_FILTERED2.xlsx'
]

# blocksize 200 filtered 110225 p70 [testing, 2013 should be the highest]
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p70_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p70_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p70_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p70_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p70_110225_FILTERED2.xlsx'
]

# blocksize 200 filtered 110225 p80 [USED]
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_110225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_110225_FILTERED2.xlsx'
]

# blocksize 200 filtered 230225 p80 with removal of edge artifacts [USED]
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'
]


#  WETTUNDRA p80 -- List of file paths for each model
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48hours.pth_MEAN_p80_280325.xlsx'
]

#  WETTUNDRA p80 -- saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425_FILTERED2.xlsx'
]

# LAKES -- p80  saved_sheets_lakesrivers_extra_filtered_ms10a9_040525
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep_MEAN_yen_280325_FILTERED2.xlsx'
]


# savedir_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/scatterplots_models_cover_means'
# shrub
savedir_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/scatterplots_allareas_models_cover_means_shrubs'
# wettundra
savedir_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_wettundra'
# lakes
savedir_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_lakes'


method_name = 'p80'


# multisitename = 'ms1'  # Change this if needed
# multisitename = 'ms2'  # Change this if needed
# multisitename = 'ms4'  # Change this if needed
# multisitename = 'ms5'  # Change this if needed
# multisitename = 'ms6'
# multisitename = 'ms7'  # Change this if needed
# multisitename = 'ms10'  # Change this if needed
# multisitename = 'ms11'  # Change this if needed
# multisitename = 'ms12'  # Change this if needed
# multisitename = 'ms13'  # Change this if needed
# multisitename = 'ms15'  # Change this if needed
multisitename = 'ms16'  # Change this if needed

# Load only the "STATISTICS_shrub_cover_analysis" sheet from each .xlsx file
data_list = [pd.read_excel(fp, sheet_name="shrub_cover_data") for fp in file_paths]
# Determine the maximum number of unique areas in any dataset to set the number of rows
max_areas = max(len(data[data["File"] == multisitename]["Area"].unique()) for data in data_list)

# Create a grid of subplots (rows = max_areas, columns = number of CSV files)
fig, axes = plt.subplots(max_areas, len(file_paths), figsize=(20, 2.5 * max_areas), sharex=True, sharey=True)
for col, (data, file_path) in enumerate(zip(data_list, file_paths)):
    modelname = file_path.split('/')[-1].split('_')[5]
    print(modelname)
    ms_data = data[data["File"] == multisitename]
    ms_areas = ms_data["Area"].unique()
    for row, area in enumerate(ms_areas):
        area_data = ms_data[ms_data["Area"] == area]
        ax = axes[row, col] if max_areas > 1 else axes[col]  # Handle single-row case
        # Determine y-axis limit
        max_y_value = max(area_data["cover_frac_binary"].max(), area_data["cover_frac_cont"].max(), 0.4)
        # Plot cover_frac_binary with error bars
        ax.errorbar(area_data["Year"], area_data["cover_frac_binary"],
                    yerr=[area_data["cover_frac_binary"] - area_data["ci_low_b"],
                          area_data["ci_up_b"] - area_data["cover_frac_binary"]],
                    fmt='o', color='red', label="cover_frac_binary", alpha=0.7)
        # Plot cover_frac_cont with error bars
        ax.errorbar(area_data["Year"], area_data["cover_frac_cont"],
                    yerr=[area_data["cover_frac_cont"] - area_data["ci_low_cont"],
                          area_data["ci_up_cont"] - area_data["cover_frac_cont"]],
                    fmt='o', color='blue', label="cover_frac_cont", alpha=0.7)
        # Calculate and plot regression line for cover_frac_binary
        if len(area_data["Year"].unique()) > 1:  # Ensure we have enough data points for regression
            z_binary = np.polyfit(area_data["Year"], area_data["cover_frac_binary"], 1)
            p_binary = np.poly1d(z_binary)
            ax.plot(area_data["Year"], p_binary(area_data["Year"]), color='red', linestyle='--',label="Regression (binary)")
            # Calculate and plot regression line for cover_frac_cont
            z_cont = np.polyfit(area_data["Year"], area_data["cover_frac_cont"], 1)
            p_cont = np.poly1d(z_cont)
            ax.plot(area_data["Year"], p_cont(area_data["Year"]), color='blue', linestyle='--', label="Regression (cont)")
        # ax.set_xticks(area_data["Year"].unique())
        # Round the x-axis ticks to the nearest integer
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(
            axis='x',
            labelrotation=45,
            labelsize=18,
            labelcolor='black',
            labelbottom=True )
        # set the y‐labels font size & weight
        ax.tick_params(
            axis='y',
            labelsize=18,
            labelcolor='black')
        # Add area name in the upper left corner
        ax.text(0.01, 0.95, f"{area}", transform=ax.transAxes, fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle="round", alpha=0.3))
        ax.set_ylim(0, max_y_value + 0.05)
        # if row == max_areas - 1:
        #     ax.set_xlabel("Year")
        # if col == 0:
        #     ax.set_ylabel("Cover Fraction")

        ax.grid(False)
        if row == 0:
            ax.set_title(f"{modelname}", fontsize=18)

fig.suptitle(f"Cover Fraction Trends for File: {multisitename}", fontsize=16)
fig.suptitle(f"\n", fontsize=16)
fig.legend(labels=["binary cover", "continuous cover"],
           # loc="upper right", bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12)
           #  loc="upper right", bbox_to_anchor=(0.93, 0.96), ncol=2, fontsize=18)
            loc="upper right", bbox_to_anchor=(1, 1), ncol=2, fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1)
fig.supxlabel("Year",            fontsize=18, fontweight="bold")
fig.supylabel("Cover Fraction\n",  fontsize=18, fontweight="bold")
# Adjust layout for better spacing
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
# Save the plot
# save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_combined_scatter_plots_FILTERED2_blocksize200_EDGEDEL_v2.jpg")
save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_SHRUB_extra_filtered__220225_FILTERED2.jpg")
# save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_wettundra_unfiltered.jpg")
# save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_wettundra_extra_filtered_ms10a9_060625_VIT.jpg")
# save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_lakes_extra_filtered_ms10a9_060625_VIT.jpg")
# plt.savefig(save_path, dpi=300)
plt.savefig(save_path)
# plt.show(block=True)
plt.close()
# plt.close('all')













import matplotlib.pyplot as plt
import numpy as np

# Data
years = [2002, 2008, 2020]
cover_binary = [0.17104773, 0.15353306, 0.18305995]
ci_low_b = [0.164714695, 0.148218606, 0.176060535]
ci_up_b = [0.177380769, 0.158847507, 0.190059357]

cover_cont = [0.09521099, 0.07383528, 0.035540834]
ci_low_cont = [0.092499636, 0.071177598, 0.034398106]
ci_up_cont = [0.097922348, 0.076492969, 0.036683562]

# 10000 iterations
# Given data
data = {
    "Year": [2005, 2013, 2020, 2017],
    "Month": [7, 8, 7, 8],
    "Day": [24, 20, 16, 13],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.212139741, 0.242420658, 0.251909971, 0.239564523],
    "cover_frac_cont": [0.229537457, 0.311624855, 0.324001461, 0.287263393],
    "ci_low_cont": [0.207174371, 0.288616609, 0.300449451, 0.264835775],
    "ci_up_cont": [0.251900544, 0.334633101, 0.347553472, 0.309691011],
    "ci_low_b": [0.186235267, 0.214981596, 0.225441554, 0.213513011],
    "ci_up_b": [0.238044214, 0.269859721, 0.278378389, 0.265616036]
}


# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]


# block 100x100
# Given data
data = {
    "Year": [2002, 2008, 2020],
    "Month": [8, 8, 6],
    "Day": [2, 17, 6],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.1710012, 0.15334731, 0.18309678],
    "cover_frac_cont": [0.09521106, 0.07375026, 0.03556504],
    "ci_low_cont": [0.088319039, 0.067145175, 0.031829526],
    "ci_up_cont": [0.102103079, 0.08035534, 0.039300556],
    "ci_low_b": [0.155513727, 0.140212714, 0.161027289],
    "ci_up_b": [0.186488665, 0.166481912, 0.205166274]
}

# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]


# Calculate errors (distance from mean to CI bounds)
error_binary = [
    np.array(cover_binary) - np.array(ci_low_b),
    np.array(ci_up_b) - np.array(cover_binary)
]

error_cont = [
    np.array(cover_cont) - np.array(ci_low_cont),
    np.array(ci_up_cont) - np.array(cover_cont)
]

# Plot
plt.figure(figsize=(10, 6))

# Plot binary cover with error bars
plt.errorbar(
    years, cover_binary, yerr=error_binary,
    fmt='o-', color='#1f77b4', label='Cover Fraction (Binary)',
    capsize=5, markersize=8
)

# Plot continuous cover with error bars
plt.errorbar(
    years, cover_cont, yerr=error_cont,
    fmt='s--', color='#ff7f0e', label='Cover Fraction (Continuous)',
    capsize=5, markersize=8
)

# Formatting
plt.xticks(years, fontsize=12)
plt.yticks(np.linspace(0, 0.4, 9), fontsize=12)
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cover Fraction', fontsize=14)
plt.title('Shrub Cover Fraction Over Time (with 95% CI)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()






# PLOT INDIVIDUAL DATA with errorbars

### BLOCK SIZE 400 x 400

import matplotlib.pyplot as plt
import numpy as np

# Given data
data = {
    "Year": [2002, 2008, 2020],
    "Month": [8, 8, 6],
    "Day": [2, 17, 6],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.1708977, 0.15354565, 0.18267071],
    "cover_frac_cont": [0.09514088, 0.07360873, 0.035626262],
    "ci_low_cont": [0.084703154, 0.063277872, 0.029447562],
    "ci_up_cont": [0.10557861, 0.083939596, 0.041804962],
    "ci_low_b": [0.148346364, 0.133529521, 0.145089234],
    "ci_up_b": [0.19344905, 0.173561775, 0.220252191]
}

# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]

# Calculate errors (distance from mean to CI bounds)
error_binary = [
    np.array(cover_binary) - np.array(ci_low_b),
    np.array(ci_up_b) - np.array(cover_binary)
]

error_cont = [
    np.array(cover_cont) - np.array(ci_low_cont),
    np.array(ci_up_cont) - np.array(cover_cont)
]

# Plot
plt.figure(figsize=(10, 6))

# Plot binary cover with error bars
plt.errorbar(
    years, cover_binary, yerr=error_binary,
    fmt='o-', color='#1f77b4', label='Cover Fraction (Binary)',
    capsize=5, markersize=8
)

# Plot continuous cover with error bars
plt.errorbar(
    years, cover_cont, yerr=error_cont,
    fmt='s--', color='#ff7f0e', label='Cover Fraction (Continuous)',
    capsize=5, markersize=8
)

# Formatting
plt.xticks(years, fontsize=12)
plt.yticks(np.linspace(0, 0.4, 9), fontsize=12)
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cover Fraction', fontsize=14)
plt.title('Shrub Cover Fraction Over Time (with 95% CI)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()




## block size 400, Iterations=10000
import matplotlib.pyplot as plt
import numpy as np

# Given data
data = {
    "Year": [2002, 2008, 2020],
    "Month": [8, 8, 6],
    "Day": [2, 17, 6],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.17115408, 0.15356933, 0.1831213],
    "cover_frac_cont": [0.095151685, 0.07374741, 0.035506368],
    "ci_low_cont": [0.084798574, 0.063685321, 0.029342113],
    "ci_up_cont": [0.105504796, 0.083809502, 0.041670623],
    "ci_low_b": [0.148876573, 0.133621646, 0.145377932],
    "ci_up_b": [0.19343159, 0.173517006, 0.220864655]
}

# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]


# Calculate errors (distance from mean to CI bounds)
error_binary = [
    np.array(cover_binary) - np.array(ci_low_b),
    np.array(ci_up_b) - np.array(cover_binary)
]

error_cont = [
    np.array(cover_cont) - np.array(ci_low_cont),
    np.array(ci_up_cont) - np.array(cover_cont)
]

# Plot
plt.figure(figsize=(10, 6))

# Plot binary cover with error bars
plt.errorbar(
    years, cover_binary, yerr=error_binary,
    fmt='o-', color='#1f77b4', label='Cover Fraction (Binary)',
    capsize=5, markersize=8
)

# Plot continuous cover with error bars
plt.errorbar(
    years, cover_cont, yerr=error_cont,
    fmt='s--', color='#ff7f0e', label='Cover Fraction (Continuous)',
    capsize=5, markersize=8
)

# Formatting
plt.xticks(years, fontsize=12)
plt.yticks(np.linspace(0, 0.4, 9), fontsize=12)
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cover Fraction', fontsize=14)
plt.title('Shrub Cover Fraction Over Time (with 95% CI)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

