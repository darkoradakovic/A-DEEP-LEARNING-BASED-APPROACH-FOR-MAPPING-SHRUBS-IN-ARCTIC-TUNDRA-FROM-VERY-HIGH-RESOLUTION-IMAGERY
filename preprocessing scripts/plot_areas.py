



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import re

from fontTools.unicodedata import block

from stats_extract_trends_excelfiles import folder_path

# Define the file path
file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx"
file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx"
file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx"
file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx"


# Load the Excel file
# df = pd.read_excel(file_path, sheet_name='with_f1_std')  # Loads all sheets into a dictionary
# df = pd.read_excel(file_path)
df = pd.read_excel(file_path, sheet_name='shrub_cover_data')  # Loads all sheets into a dictionary

# shrubs
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'
]

# Wettundra
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48hours.pth_MEAN_p80_280325_FILTERED2.xlsx'
]

data_list = [pd.read_excel(fp, sheet_name="shrub_cover_data") for fp in file_paths]

multisitename = 'ms6'
max_areas = max(len(data[data["File"] == multisitename]["Area"].unique()) for data in data_list)



# ### --- Plot all areas of one Site binary mean cover
#
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Filter for file 'ms1'
# df_ms1 = df[df['File'] == 'ms1']
#
# # Group by Area and calculate the mean values for cover fraction and CI bounds.
# grouped = df_ms1.groupby('Area').agg({
#     'cover_frac_binary': 'mean',
#     'ci_low_b': 'mean',
#     'ci_up_b': 'mean'
# }).reset_index()
#
# # Compute the error bars:
# # Error below = mean - ci_low, and error above = ci_up - mean.
# y = grouped['cover_frac_binary']
# error_lower = y - grouped['ci_low_b']
# error_upper = grouped['ci_up_b'] - y
# yerr = [error_lower, error_upper]
#
# # For the x-axis, we can use a simple range of numbers corresponding to each area.
# x = range(len(grouped))
#
# plt.figure(figsize=(10, 6))
# plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, ecolor='black', markerfacecolor='blue', markersize=8)
# plt.xticks(x, grouped['Area'], rotation=45, fontsize=12)
# plt.xlabel('Area', fontsize=14)
# plt.ylabel('Mean Cover Fraction Binary', fontsize=14)
# plt.title("Mean Cover Fraction Binary with 95% Confidence Intervals (ms1)", fontsize=16)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#

# ### --- Plot all areas of one Site Continuous mean cover
#
# df_ms1 = df[df['File'] == 'ms1']
#
# # Group by Area and calculate the mean values for cover fraction and CI bounds.
# grouped = df_ms1.groupby('Area').agg({
#     'cover_frac_cont': 'mean',
#     'ci_low_cont': 'mean',
#     'ci_up_cont': 'mean'
# }).reset_index()
#
# # Compute the error bars:
# # Error below = mean - ci_low, and error above = ci_up - mean.
# y = grouped['cover_frac_cont']
# error_lower = y - grouped['ci_low_cont']
# error_upper = grouped['ci_up_cont'] - y
# yerr = [error_lower, error_upper]
#
# # For the x-axis, we can use a simple range of numbers corresponding to each area.
# x = range(len(grouped))
#
# plt.figure(figsize=(10, 6))
# plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, ecolor='black', markerfacecolor='blue', markersize=8)
# plt.xticks(x, grouped['Area'], rotation=45, fontsize=12)
# plt.xlabel('Area', fontsize=14)
# plt.ylabel('Mean Cover Fraction Continuous', fontsize=14)
# plt.title("Mean Cover Fraction Continuous with 95% Confidence Intervals (ms1)", fontsize=16)
# plt.grid(True)
# plt.tight_layout()
# plt.show()



### BARPLOT SHURB COVER ALL MODELS PER SITES
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# List of file paths for each model - SHRUBS
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'
]

# List of file paths for each model - WETTUNDRA
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425.xlsx'
]

# LAKES
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers/STATISTICS_shrub_cover_analysis_lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep_MEAN_otsu_280325.xlsx'
]

# Define your selected models (order corresponds to file_paths)
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
# selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil"]
# Read each Excel file into a DataFrame and add a "Model" column.
data_list = []
for model, fp in zip(selected_models, file_paths):
    df_model = pd.read_excel(fp, sheet_name="shrub_cover_data")
    df_model["Model"] = model
    data_list.append(df_model)

# Concatenate all DataFrames
df_all = pd.concat(data_list, ignore_index=True)

# Optionally, standardize the File column. For example, replace 'ms' with 'site'
df_all["File"] = df_all["File"].str.replace("ms", "site", regex=False)

# Optionally, sort sites by numeric order extracted from the File string.
def extract_num(s):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0

sites = sorted(df_all["File"].unique(), key=lambda x: extract_num(x))
n_sites = len(sites)
n_models = len(selected_models)

# Pivot the DataFrame so that the index is File and columns are multi-indexed by metric and Model.
# We assume the columns are: "cover_frac_cont", "ci_low_cont", "ci_up_cont"
pivot_df = pd.pivot_table(
    df_all,
    index="File",
    columns="Model",
    values=["cover_frac_cont", "ci_low_cont", "ci_up_cont"],
    aggfunc='mean'
)

# Ensure the pivot has the sites in the desired order:
pivot_df = pivot_df.reindex(sites)
# Define bar width and x positions (one group per site)
bar_width = 0.15
x = np.arange(n_sites)

fig, ax = plt.subplots(figsize=(12, 8))
for i, model in enumerate(selected_models):
    # Extract mean cover fraction and confidence intervals for this model
    cover = pivot_df["cover_frac_cont"][model]
    ci_lower = pivot_df["ci_low_cont"][model]
    ci_upper = pivot_df["ci_up_cont"][model]
    # Calculate error bars as half the CI width.
    err_lower = cover - ci_lower
    err_upper = ci_upper - cover
    yerr = [err_lower, err_upper]
    # Compute x positions for each model's bar in a group (offset each model)
    offset = (i - (n_models - 1) / 2) * bar_width
    ax.bar(x + offset, cover, width=bar_width, yerr=yerr, capsize=5, label=model, alpha=0.8)

# Set the x-axis ticks and labels to the site names.
ax.set_xticks(x)
ax.set_xticklabels(sites, rotation=45, fontsize=20, fontweight='bold')
ax.set_xlabel("Site", fontsize=20, fontweight='bold')
# ax.set_ylabel("Mean Shrub Cover \nFraction (Continuous)", fontsize=20, fontweight='bold')
ax.set_ylabel("Mean Wet Tundra Cover \nFraction (Continuous)", fontsize=20, fontweight='bold')
# ax.set_ylabel("Mean Surface Water Bodies \n Cover Fraction (Continuous)", fontsize=20, fontweight='bold')
# ax.set_title("Mean Tall Shrub Cover Fraction (Continuous) per Site\nwith 95% Confidence Intervals by Model", fontsize=20, fontweight='bold')
# ax.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
ax.legend(title="Model", fontsize=20, title_fontsize=20)
plt.tight_layout()
plt.show(block=True)


## stats COVER vs site

import re
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
import itertools

# (1) Read each model’s file and concatenate into one DataFrame (you’ve already done this).
#     Here we assume `df_all` already exists.
#     If not, you’d do something like:
#
# selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
# data_list = []
# for model, fp in zip(selected_models, file_paths):
#     df_model = pd.read_excel(fp, sheet_name="shrub_cover_data")
#     df_model["Model"] = model
#     data_list.append(df_model)
# df_all = pd.concat(data_list, ignore_index=True)
#
# (2) Standardize your site names if needed; e.g. replace "ms" with "site" so they sort nicely.
df_all["File"] = df_all["File"].str.replace("ms", "site", regex=False)

# (3) Ensure `cover_frac_cont` is a float column; drop any NaNs in that column:
df_all = df_all.dropna(subset=["cover_frac_cont"])

# (4) Pivot the data so that each row = one site, each column = one model’s cover_frac_cont.
#     We take the mean (in case a given site–model combination appears more than once).
pivot_cover = df_all.pivot_table(
    index="File",
    columns="Model",
    values="cover_frac_cont",
    aggfunc="mean"
)

# (5) Sort the sites in numeric order (e.g. site1, site2, …). This step is optional,
#     but it helps keep the table in a logical sequence:
def extract_num(s):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

pivot_cover = pivot_cover.reindex(
    sorted(pivot_cover.index, key=extract_num)
)

# Now `pivot_cover` has shape (n_sites, 5), like:
#          CNN 512fil  Resnet50  VGG19  UNET 256fil   ViT
# site1        0.30      0.29    0.31       0.27    0.24
# site2        0.14      0.13    0.18       0.16    0.19
# ...
# siteN        0.12      0.15    0.16       0.14    0.18

# -------------------------------------------------------------------------------
# 6) Check assumptions across “sites” (rows) for a repeated‐measures ANOVA:
#    We need to know if the residuals across the five “Model” columns are (approximately) normal
#    and if the sphericity assumption holds. In practice, sphericity is often violated,
#    so a safer route is to run Friedman’s nonparametric test.
# -------------------------------------------------------------------------------

# (6a) Friedman test (nonparametric, one‐way, repeated‐measures)
#     Inputs: a 2D array of shape (n_sites, n_models), i.e. each row is one “subject” (site),
#     and each column is one repeated “treatment” (model).
data_matrix = pivot_cover.values  # shape = (n_sites, 5)

friedman_stat, friedman_p = stats.friedmanchisquare(
    *(data_matrix[:, i] for i in range(data_matrix.shape[1]))
)
print(f"Friedman test: χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}")

# If p < 0.05, we reject the null hypothesis that all five model‐means (across sites) are identical.
# That tells us that at least one “site” differs in median cover from others (when looking across the five models).

# -------------------------------------------------------------------------------
# 7) Post‐hoc pairwise tests: Dunn’s test (nonparametric) to compare each pair of sites.
#    We need to “unroll” the pivot matrix into a long format for Dunn’s test.
# -------------------------------------------------------------------------------

# Build a DataFrame suitable for posthoc_dunn:
# We’ll create three columns: “Site”, “Model”, “Cover_frac”
long_list = []
for site in pivot_cover.index:
    for model in pivot_cover.columns:
        cover_val = pivot_cover.loc[site, model]
        long_list.append((site, model, cover_val))
long_df = pd.DataFrame(long_list, columns=["Site", "Model", "Cover_frac"])

# Now we run posthoc Dunn’s test (group_col = "Site", val_col = "Cover_frac").
dunn_res = sp.posthoc_dunn(
    long_df,
    val_col="Cover_frac",
    group_col="Site",
    p_adjust="bonferroni"   # or “holm”, “fdr_bh”, etc.
)

# `dunn_res` is a DataFrame of size (n_sites × n_sites), giving pairwise adjusted p‐values.
print("Pairwise Dunn’s test (cover_frac_cont) among sites (Bonferroni‐adjusted):")
print(dunn_res.round(4))

# You can then inspect which site‐pairs have p < 0.05, meaning their distributions of cover (across the
# five models) differ significantly from each other.












### TRENDS FOR ALL MODELS
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re



# SHRUB file paths for each model
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'
]

# Shrub bootstrap CI (removed?)
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_270225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_270225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_270225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_270225.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_270225.xlsx'
]

# List of file paths for each model - WETTUNDRA
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48hours.pth_MEAN_p80_280325.xlsx'
]

# LAKES
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers/STATISTICS_shrub_cover_analysis_lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep_MEAN_otsu_280325.xlsx'
]

# Define your selected models (order corresponds to file_paths)
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil"]

# Read each Excel file into a DataFrame and add a "Model" column.
data_list = []
for model, fp in zip(selected_models, file_paths):
    df_model = pd.read_excel(fp, sheet_name="shrub_cover_statistics")
    df_model["Model"] = model
    data_list.append(df_model)

# Concatenate all DataFrames
df_all = []
df_all = pd.concat(data_list, ignore_index=True)

# (Optional) Standardize the File column to use "site" instead of "ms"
df_all["File"] = df_all["File"].str.replace("ms", "site", regex=False)

# Pivot the DataFrame to get separate columns for each model.
# In case of duplicate File/Model combinations, we take the mean.
pivot_df = pd.pivot_table(
    df_all,
    index="File",
    columns="Model",
    values=["Slope Continuous", "CI Lower Continuous", "CI Upper Continuous"],
    aggfunc='mean'
)


# Ensure the sites (File) are sorted in numerical order.
def extract_num(s):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0


sites = sorted(pivot_df.index, key=extract_num)
n_sites = len(sites)
n_models = len(selected_models)

# Create x positions for each site group.
x = np.arange(n_sites)
bar_width = 0.20

fig, ax = plt.subplots(figsize=(12, 8))
# For each model, plot the corresponding bar with error bars.
for i, model in enumerate(selected_models):
    # Retrieve the cover fraction and CI values for this model.
    cover = pivot_df["Slope Continuous"][model].reindex(sites)
    ci_lower = pivot_df["CI Lower Continuous"][model].reindex(sites)
    ci_upper = pivot_df["CI Upper Continuous"][model].reindex(sites)

    # Calculate error as half the CI range.
    err_lower = cover - ci_lower
    err_upper = ci_upper - cover
    yerr = [err_lower, err_upper]

    # Compute x positions for the bars of this model.
    offset = (i - (n_models - 1) / 2) * bar_width
    ax.bar(x + offset, cover, width=bar_width, yerr=yerr, capsize=5,
           label=model, alpha=0.8)

# Set x-axis tick labels to the site names.
ax.set_xticks(x)
ax.set_xticklabels(sites, rotation=45, fontsize=20, fontweight='bold')
ax.set_xlabel("Site", fontsize=20, fontweight='bold')
# ax.set_ylabel("Mean Tall Shrub Cover Trends \n(Continuous) (%)", fontsize=20, fontweight='bold')
# ax.set_ylabel("Mean Wet Tundra Cover Trends \n(Continuous) (%)", fontsize=20, fontweight='bold')
ax.set_ylabel("Mean Surface Water Bodies \nCover Trends (Continuous) (%)", fontsize=20, fontweight='bold')
# ax.set_title("Mean Tall Shrub Cover Trends per Site\nwith 95% Confidence Intervals by Model", fontsize=20, fontweight='bold')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
# ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(title="Model", fontsize=20, title_fontsize=20)
plt.tight_layout()
plt.show(block=True)










### --- Alternative Plot all sites with error bars for mean cover (ONE MODEL)
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

grouped = df.groupby('File').agg({
    'cover_frac_binary': 'mean',
    'ci_low_b': 'mean',
    'ci_up_b': 'mean'
}).reset_index()

grouped['File'] = grouped['File'].str.replace('ms', 'site')
grouped = grouped.sort_values(by='File', key=lambda x: x.str.extract('(\d+)').astype(int)[0])

# Calculate error bars.
# Error below: difference between mean cover and lower CI
# Error above: difference between upper CI and mean cover.
mean_cover = grouped['cover_frac_binary']
err_lower = mean_cover - grouped['ci_low_b']
err_upper = grouped['ci_up_b'] - mean_cover
yerr = [err_lower, err_upper]

# Create x positions for each File.
x = range(len(grouped))


plt.figure(figsize=(10, 6))
plt.errorbar(x, mean_cover, yerr=yerr, fmt='o', capsize=5, color='blue', markerfacecolor='white', markersize=10)
plt.xticks(x, grouped['File'], rotation=45, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('File', fontsize=16, fontweight='bold')
plt.ylabel('Mean Cover Fraction Binary', fontsize=16, fontweight='bold')
plt.title('Mean Cover Fraction Binary per Site with 95% Confidence Intervals', fontsize=18, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show(block=True)




## Continuous  (ONE MODEL)

grouped = df.groupby('File').agg({
    'cover_frac_cont': 'mean',
    'ci_low_cont': 'mean',
    'ci_up_cont': 'mean'
}).reset_index()

grouped['File'] = grouped['File'].str.replace('ms', 'site')
grouped = grouped.sort_values(by='File', key=lambda x: x.str.extract('(\d+)').astype(int)[0])

# Calculate error bars.
# Error below: difference between mean cover and lower CI
# Error above: difference between upper CI and mean cover.
mean_cover = grouped['cover_frac_cont']
err_lower = mean_cover - grouped['ci_low_cont']
err_upper = grouped['ci_up_cont'] - mean_cover
yerr = [err_lower, err_upper]

# Create x positions for each File.
x = range(len(grouped))


plt.figure(figsize=(10, 6))
plt.errorbar(x, mean_cover, yerr=yerr, fmt='o', capsize=5, color='blue', markerfacecolor='white', markersize=10)
plt.xticks(x, grouped['File'], rotation=45, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('File', fontsize=16, fontweight='bold')
plt.ylabel('Mean Cover Fraction Continuous', fontsize=16, fontweight='bold')
# plt.title('Mean Cover Fraction Continuous per Site with 95% \n Confidence Intervals for CNN 512fil', fontsize=18, fontweight='bold')
plt.title('Mean Cover Fraction Continuous per Site with 95% \n Confidence Intervals for VGG19', fontsize=18, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show(block=True)




### --- --- Plot cover PER AREA, all sites FIRST PART

# ### SHRUB -- saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9
# # CNN
# modelname = 'CNN 512-filters'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx'
# # ResNet50
# modelname = 'ResNet50'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx'
# # VGG19
# modelname = 'VGG19'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx'
# # U-Net
# modelname = 'U-Net'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx'
# # VIT
# modelname = 'VIT'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'
#
# ### Wettundra
# #  WETTUNDRA p80 -- saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT
#
# # CNN
# modelname = 'CNN 512-filters'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx'
# # ResNet50
# modelname = 'ResNet50'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx'
# # VGG19
# modelname = 'VGG19'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx'
# # U-Net
# modelname = 'U-Net'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx'
# # VIT
# modelname = 'VIT'
# file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425_FILTERED2.xlsx'
#
# wet_files = [
#     ("CNN", '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx'),
#     ("ResNet50",   '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx'),
#     ("VGG19",      '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx'),
#     ("U-Net",'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx'),
#     ("VIT",        '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425_FILTERED2.xlsx'),
# ]
# # wettundra save
# outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_wettundra'
# # # lakes
# # outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_lakes'
#
# method_name = 'p80'
#
# df = pd.read_excel(file_path, sheet_name='shrub_cover_data')  # Loads all sheets into a dictionary
#
# df['File'] = df['File'].str.replace('ms', 'site')
#
# # Extract numeric parts for sorting.
# df['site_num'] = df['File'].str.extract('(\d+)').astype(int)
# df['area_num'] = df['Area'].str.extract('(\d+)').astype(int)
#
# # Group by File and Area, then calculate the mean values.
# df_grouped = df.groupby(['File', 'Area', 'site_num', 'area_num']).agg({
#     'cover_frac_cont': 'mean',
#     'ci_low_cont': 'mean',
#     'ci_up_cont': 'mean'
# }).reset_index()
#
# # # Select only the first 4 sites (using site_num for sorting).
# df_grouped = df_grouped.sort_values(by=['site_num', 'area_num'])
# unique_sites = sorted(df_grouped['File'].unique(), key=lambda x: int(x.replace('site','')))
# selected_sites = unique_sites[:6]
# df_plot = df_grouped[df_grouped['File'].isin(selected_sites)].copy()
#
# df_grouped = df_grouped.sort_values(by=['site_num', 'area_num'])
# # df_grouped = df.sort_values(by=['site_num', 'area_num'])
#
# # Select the sites you want to plot (for example, first 4 sites).
# unique_sites = sorted(df_grouped['File'].unique(), key=lambda x: int(x.replace('site', '')))
# selected_sites = unique_sites[:6]
# df_plot = df_grouped[df_grouped['File'].isin(selected_sites)].copy()
#
# # --- Assign x positions sequentially ---
# # Define gaps: gap between consecutive areas and gap between sites.
# gap_intra = 0.1  # gap between areas within a site
# gap_inter = 0.3  # gap between sites
#
# # We'll assign x positions in a continuous manner.
# x_positions = []
# labels = []
# current_x = 0
#
# # Process each selected site in order.
# for site in selected_sites:
#     sub = df_plot[df_plot['File'] == site].copy().sort_values(by='area_num')
#     n = len(sub)
#     # Assign x positions for the areas in this site.
#     positions = current_x + np.arange(n) * gap_intra
#     # Record the positions for each row.
#     df_plot.loc[df_plot['File'] == site, 'x'] = positions
#     # Create labels (you can change how you want to label each point).
#     for pos, area in zip(positions, sub['Area']):
#         labels.append(f"{site}-{area}")
#     # Update current_x for the next site: add the last position plus a gap between sites.
#     current_x = positions[-1] + gap_inter
#
# # --- Calculate error bars ---
# mean_cover = df_plot['cover_frac_cont']
# err_lower = mean_cover - df_plot['ci_low_cont']
# err_upper = df_plot['ci_up_cont'] - mean_cover
# yerr = [err_lower, err_upper]
#
# # --- Plot ---
# plt.figure(figsize=(16, 6))
# plt.errorbar(df_plot['x'], mean_cover, yerr=yerr, fmt='o', capsize=5,
#              color='blue', markerfacecolor='white', markersize=8)
# plt.xticks(df_plot['x'], df_plot.apply(lambda row: f"{row['File']}-{row['Area']}", axis=1), rotation=45, fontsize=8, fontweight='bold')
# plt.xlabel('Site - Area', fontsize=18, fontweight='bold')
# plt.ylabel('Mean Cover Fraction \nContinuous', fontsize=20, fontweight='bold')
# # plt.title(f'Mean Cover Fraction (Continuous Predictions) per Area (Grouped by Site)\n for model {modelname} with 95% CI intervals with site 1 - 7', fontsize=20, fontweight='bold')
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.ylim(0, 1)
# plt.grid(True)
# plt.tight_layout()
# plt.show(block=True)
#
# save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_wettundra_extra_filtered_ms10a9_060625_VIT_firsthalf_COVER.jpg")
# # save_path = os.path.join(savedir_path, f"{multisitename+'_'+method_name}_lakes_extra_filtered_ms10a9_060625_VIT.jpg")
# # plt.savefig(save_path, dpi=300)
# plt.savefig(save_path)
# plt.close()
#
#
#
#
#
# ## second half




### ### ---------- APPENDIX ---------- SAVE BOTH HALVES LOOP

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

shrub_files = [
    ("CNN", '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx'),
    ("ResNet50",   '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx'),
    ("VGG19",      '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx'),
    ("U-Net",   '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx'),
    ("VIT",        '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'),
]


wet_files = [
    ("CNN", '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx'),
    ("ResNet50",   '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx'),
    ("VGG19",      '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx'),
    ("U-Net",'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx'),
    ("VIT",        '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425_FILTERED2.xlsx'),
]
# wettundra save
outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_wettundra'

lake_files = [
    ("CNN", '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx'),
    ("ResNet50",   '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx'),
    ("VGG19",      '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_p80_280325_FILTERED2.xlsx'),
    ("U-Net",'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx'),
    ("VIT",        '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep_MEAN_yen_280325_FILTERED2.xlsx'),
]


# # lakes
outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/scatterplots_allareas_models_cover_means_shrubs'
# outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_lakes'

if not os.path.exists(outdir):
    os.mkdir(outdir)

def plot_sites_two_halves(model_name, filepath, target, outdir):
    # 1) Read data & rename
    df = pd.read_excel(filepath, sheet_name="shrub_cover_data")
    df["File"] = df["File"].str.replace("ms", "site", regex=False)
    df["site_num"] = df["File"].str.extract(r"(\d+)").astype(int)
    df["area_num"] = df["Area"].str.extract(r"(\d+)").astype(int)

    # 2) Aggregate
    dfg = (
        df.groupby(["File","Area","site_num","area_num"])
          .agg(cover=("cover_frac_cont","mean"),
               low=("ci_low_cont","mean"),
               high=("ci_up_cont","mean"))
          .reset_index()
          .sort_values(["site_num","area_num"])
    )

    # 3) Split sites
    sites = sorted(dfg["File"].unique(), key=lambda x: int(re.search(r"\d+", x).group()))
    mid = len(sites)//2
    halves = [sites[:mid], sites[mid:]]

    # 4) Plot each half
    for idx, half_sites in enumerate(halves, start=1):
        sub = dfg[dfg["File"].isin(half_sites)]
        xs, ys, yerr_low, yerr_high, labels = [], [], [], [], []
        cx = 0
        for site in half_sites:
            sdf = sub[sub["File"] == site]
            sdf = sdf.sort_values("area_num")
            n = len(sdf)
            xloc = cx + np.arange(n)*0.1
            xs.extend(xloc)
            ys.extend(sdf["cover"])
            yerr_low.extend(sdf["cover"] - sdf["low"])
            yerr_high.extend(sdf["high"] - sdf["cover"])
            labels.extend(f"{site}-{a}" for a in sdf["Area"])
            cx = xloc[-1] + 0.3

        fig, ax = plt.subplots(figsize=(16,6))
        ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], fmt="o", capsize=5, color="blue")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Site–Area", fontsize=18, fontweight="bold")
        ax.set_ylabel("Cover Fraction (cont.)", fontsize=18, fontweight="bold")
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.ylim(0, 1)
        # ax.set_title(f"{model_name} {target} – Sites Half {idx}", fontsize=14)
        ax.grid(False)
        plt.tight_layout()
        fn = f"{model_name}_{target}_sites_half{idx}_COVER.jpg"
        fig.savefig(os.path.join(outdir, fn), dpi=200)
        plt.close(fig)

# Generate plots
for model_name, fp in shrub_files:
    plot_sites_two_halves(model_name, fp, "SHRUBS", outdir)


for model_name, fp in wet_files:
    plot_sites_two_halves(model_name, fp, "WetTundra", outdir)

for model_name, fp in lake_files:
    plot_sites_two_halves(model_name, fp, "LAKES", outdir)












### --- --- --- TRENDS


### TRENDS TABLE
# shrub
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/TRENDS_WEIGHTED.xlsx'
# Wet tundra
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/TRENDS_WEIGHTED_wettundra.xlsx'
# Lakes
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/TRENDS_WEIGHTED_lakes.xlsx'



#######  TRENDS PER METHOD BINARY
# NO need to drop, already using only weighted values

df3 = pd.read_excel(folder_path)  # Loads all sheets into a dictionary
pivot_mean = df3.pivot_table(index='Method', columns='Model', values='Slope_Binary', aggfunc='mean')
pivot_p = df3.pivot_table(index='Method', columns='Model', values='p_value_binary', aggfunc='mean')


def stars_from_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# Get sorted lists of thresholding methods and models.
methods = pivot_mean.index.tolist()
models = pivot_mean.columns.tolist()
# models = ['CNN_512fil', 'ResNet50', 'U-Net_256fil', 'VGG19']

# Create a dictionary mapping each model to a unique color.
model_colors = {
    "ResNet50": "red",
    "U-Net_256fil": "blue",
    "VGG19": "green",
    "CNN_512fil": "purple",
    "VIT": "orange"
}
# If your model names in your DataFrame differ slightly, adjust the keys accordingly.

# Set up positions for grouped bars.
x = np.arange(len(methods))
n_models = len(models)
width = 0.8 / n_models  # total width for each group

fig, ax = plt.subplots(figsize=(12, 6))
# Plot bars for each model using its assigned color.
for i, model in enumerate(models):
    values = pivot_mean[model].values*100
    color = model_colors.get(model, "skyblue")  # fallback to skyblue if model not in dictionary
    bars = ax.bar(x + i * width, values, width, label=model, color=color, edgecolor='black')

    # Annotate each bar with significance stars.
    for j, bar in enumerate(bars):
        p_val = pivot_p[model].iloc[j]
        star_text = stars_from_p(p_val)
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.0005, star_text,
                ha='center', va='bottom', fontsize=12, color='black') # color='red'

# Set x-ticks so that they appear centered for each group.
ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels(methods, rotation=45, fontsize=18, fontweight='bold')

ax.set_xlabel("Thresholding Method", fontsize=18, fontweight='bold')
ax.set_ylabel("Weighted Trend \n(Binary) (%)", fontsize=18, fontweight='bold')
# ax.set_title("Weighted Trend per Thresholding Method by Model\nwith Significance Annotation",
#              fontsize=20, fontweight='bold')
ax.legend(title="Model", fontsize=18, title_fontsize=18)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show(block=True)




#### TRENDS CONT (WEIGHTED)

df_model = df3.groupby("Model").agg({
    "Slope_Continuous": "mean",
    "p_value_cont": "mean"
}).reset_index()

# Convert slope to percentages (if desired)
df_model["Slope_Continuous_pct"] = df_model["Slope_Continuous"] * 100

# Define colors for each model.
model_colors = {
    "ResNet50": "red",
    "U-Net_256fil": "blue",
    "VGG19": "green",
    "CNN_512fil": "purple",
    "VIT": "orange"
}

# models = ['CNN_512fil', 'ResNet50', 'U-Net_256fil', 'VGG19']

# Define a function to convert p-values to significance stars.
def stars_from_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# Create x positions for the models.
x = np.arange(len(df_model))
width = 0.6  # width of each bar


fig, ax = plt.subplots(figsize=(6, 6))
# Create bars for each model.
bars = ax.bar(x, df_model["Slope_Continuous_pct"], width,
              color=[model_colors.get(model, "gray") for model in df_model["Model"]],
              edgecolor='black')
# Annotate each bar with significance stars.
for i, bar in enumerate(bars):
    p_val = df_model.loc[i, "p_value_cont"]
    star_text = stars_from_p(p_val)
    height = bar.get_height()
    # Adjust the vertical offset as needed.
    # ax.text(bar.get_x() + bar.get_width() / 2, height + 0.0005, star_text,  # shrub
    # ax.text(bar.get_x() + bar.get_width() / 2, height - 0.0255, star_text,  # wettundra
    ax.text(bar.get_x() + bar.get_width() / 2, height - 0.0045, star_text,  # lakes
            ha='center', va='bottom', fontsize=12, color='black')  # color='red'
# Set x-axis labels.
ax.set_xticks(x)
ax.set_xticklabels(df_model["Model"], fontsize=18, fontweight='bold', rotation=35, ha='right')
ax.set_xlabel("Models", fontsize=18, fontweight='bold')
ax.set_ylabel("Weighted Continuous \nTrend (%)", fontsize=18, fontweight='bold')
# ax.set_ylabel("Weighted Continuous Trend\n Shrub (%)", fontsize=16, fontweight='bold')
# ax.set_ylabel("Weighted Continuous Trend\n Wet Tundra Cover (%)", fontsize=20, fontweight='bold')
# ax.set_ylabel("Weighted Continuous Trend\n surface water bodies Cover (%)", fontsize=20, fontweight='bold')
# ax.set_title("Weighted Continuous Trend \nby Model", fontsize=20, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
# plt.ylim(-0.1, 0.02)  # FOR WETTUNDRA
plt.tight_layout()
# plt.show(block=True)
plt.show()




#### BARPLOT SHRUB WET TUNDRA + LAKES in one
df_shrub = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/TRENDS_WEIGHTED.xlsx')
df_wet   = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/TRENDS_WEIGHTED_wettundra.xlsx')
df_lake  = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/TRENDS_WEIGHTED_lakes.xlsx')

# 2) tag them
df_shrub["Type"] = "Shrub"
df_wet  ["Type"] = "WetTundra"
df_lake ["Type"] = "Lake"

# 3) stack
df_all = pd.concat([df_shrub, df_wet, df_lake], ignore_index=True)

# 4) pull off the continuous slopes (strip “%” if present) as floats
df_all["Slope_pct"] = (
    df_all["Slope_Continuous"]
      .astype(str)
      .str.rstrip("%")
      .astype(float)
)
# 5) pivot to get one row per model, columns for each Type
df_pivot = df_all.pivot_table(
    index="Model",
    columns="Type",
    values="Slope_pct",
    aggfunc="mean"
)
# 5b) pivot p‐values similarly
pv = df_all.pivot_table(
    index="Model",
    columns="Type",
    values="p_value_cont",
    aggfunc="mean"
)
# 6) force the column order you want:
desired = ["Shrub","WetTundra","Lake"]
df_pivot = df_pivot[desired]
pv       = pv[desired]


# 7) plot
fig, ax = plt.subplots(figsize=(8,8))
df_pivot.plot.bar(
    ax=ax, width=0.8, edgecolor="k",
    # ylabel="Weighted Continuous Trend (%)",
    fontsize=16,
    # title ="Annual Trends by Model & Feature"
)
ax.set_xlabel("Models", fontsize=16, fontweight='bold')
# ax.legend(title="Feature", fontsize=16)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
# 8) significance‐star helper
def stars(p):
    if p<0.001: return "***"
    if p<0.01:  return "**"
    if p<0.05:  return "*"
    return ""         # skip “ns”

# 9) annotate each bar
n_types = len(desired)
for i, model in enumerate(df_pivot.index):
    for j, feat in enumerate(desired):
        bar_idx = i*n_types + j
        bar     = ax.patches[bar_idx]
        pval    = pv.loc[model, feat]
        star_txt = stars(pval)
        if star_txt:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0003,    # tweak vertical offset
                star_txt,
                ha="center", color="red", fontsize=12
            )
ax.set_xlabel("Models", fontsize=16, fontweight='bold')
ax.set_ylabel("Weighted Continuous Trends", fontsize=16, fontweight='bold')
plt.legend(fontsize=14, loc="best", labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show(block=True)







### TRENDS FOR ONE model
df2 = pd.read_excel(file_path, sheet_name='shrub_cover_statistics')  # Loads all sheets into a dictionary

df2['File'] = df2['File'].str.replace('ms', 'site')

# Ensure proper numerical order of Files (assuming names like site1, site2, ...)
df2 = df2.sort_values(by='File', key=lambda x: x.str.extract('(\d+)').astype(int)[0])

df2 = df2[np.isfinite(df2['Slope Continuous']) &
          np.isfinite(df2['CI Lower Continuous']) &
          np.isfinite(df2['CI Upper Continuous'])]

df2 = df2.dropna(subset=['Slope Continuous', 'CI Lower Continuous', 'CI Upper Continuous'])


# Group by File and compute the mean trend (slope) and its confidence interval bounds.
grouped = df2.groupby('File').agg({
    'Slope Continuous': 'mean',
    'CI Lower Continuous': 'mean',
    'CI Upper Continuous': 'mean'
}).reset_index()

grouped = grouped.sort_values(by='File', key=lambda x: x.str.extract('(\d+)').astype(int)[0])


# Calculate error bars:
#   The error below is the difference between the mean slope and the lower CI.
#   The error above is the difference between the upper CI and the mean slope.
mean_slope = grouped['Slope Continuous']
err_lower = mean_slope - grouped['CI Lower Continuous']
err_upper = grouped['CI Upper Continuous'] - mean_slope
yerr = [err_lower, err_upper]

# Create x positions (one per site)
x = range(len(grouped))

# Plot the trends (average slopes) with error bars.
plt.figure(figsize=(10, 6))
plt.errorbar(x, mean_slope, yerr=yerr, fmt='o', capsize=5,
             color='blue', markerfacecolor='white', markersize=10)
plt.xticks(x, grouped['File'], rotation=45, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('Site', fontsize=16, fontweight='bold')
plt.ylabel('Tall Shrub Cover Trend (%)', fontsize=16, fontweight='bold')
plt.title('Average Trend Tall Shrub Cover (Continuous predictions) \n per Site with 95% CI for CNN 512fil', fontsize=18, fontweight='bold')
# plt.title('Average Trend Tall Shrub Cover (Continuous predictions) \n per Site with 95% CI for VGG19', fontsize=18, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show(block=True)







### ALL AREAS TREND FIRST PART
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppose your Excel file is loaded in df2 (from 'shrub_cover_statistics' sheet).
# Columns of interest:
#   'File', 'Area', 'Slope Continuous', 'CI Lower Continuous', 'CI Upper Continuous'
#   plus any numeric extraction for sorting: site_num, area_num

# 1. Clean up 'File' column and parse numeric parts for sorting
df2['File'] = df2['File'].str.replace('ms', 'site')
df2['site_num'] = df2['File'].str.extract(r'(\d+)').astype(int)
df2['area_num'] = df2['Area'].str.extract(r'(\d+)').astype(int)

# 2. Remove any rows with non-finite slope or CI
df2 = df2[np.isfinite(df2['Slope Continuous']) &
          np.isfinite(df2['CI Lower Continuous']) &
          np.isfinite(df2['CI Upper Continuous'])]

# 3. Group by (File, Area) to get a single slope & CI for each site-area pair.
#    (If each site-area-year has a slope, you might average them here, or
#     if they are already aggregated, skip this step.)
df_grouped = df2.groupby(['File', 'Area', 'site_num', 'area_num'], dropna=False).agg({
    'Slope Continuous': 'mean',
    'CI Lower Continuous': 'mean',
    'CI Upper Continuous': 'mean'
}).reset_index()

# 4. Sort by site_num, then by area_num for a logical x-axis order
df_grouped = df_grouped.sort_values(by=['site_num', 'area_num'])

# 5. (Optional) If you only want the first N sites:
unique_sites = sorted(df_grouped['File'].unique(), key=lambda x: int(x.replace('site', '')))
selected_sites = unique_sites[:6]  # for example, first 6 sites
df_plot = df_grouped[df_grouped['File'].isin(selected_sites)].copy()

# 6. Assign x positions in a continuous manner, grouping by site
gap_intra = 0.1  # gap between areas within a site
gap_inter = 0.3  # gap between consecutive sites
x_positions = []
labels = []
current_x = 0

for site in selected_sites:
    sub = df_plot[df_plot['File'] == site].copy().sort_values(by='area_num')
    n = len(sub)
    # Assign x positions for the areas in this site
    positions = current_x + np.arange(n) * gap_intra
    # Store positions
    df_plot.loc[sub.index, 'x'] = positions
    # Create labels
    for pos, area in zip(positions, sub['Area']):
        labels.append(f"{site}-{area}")
    # Advance current_x for the next site
    current_x = positions[-1] + gap_inter

# 7. Calculate error bars from slope and CI
mean_slope = df_plot['Slope Continuous']
err_lower = mean_slope - df_plot['CI Lower Continuous']
err_upper = df_plot['CI Upper Continuous'] - mean_slope
yerr = [err_lower, err_upper]

# 8. Plot
plt.figure(figsize=(16, 6))
plt.errorbar(df_plot['x'], mean_slope, yerr=yerr,
             fmt='o', capsize=5, color='blue',
             markerfacecolor='white', markersize=8)
plt.xticks(df_plot['x'], labels, rotation=45, fontsize=8, fontweight='bold')
plt.xlabel('Site - Area', fontsize=10, fontweight='bold')
plt.ylabel('Slope of Tall Shrub Cover Trend', fontsize=10, fontweight='bold')
plt.title('Per-Area Trends in Tall Shrub Cover (Continuous Predictions) by Site for CNN 512fil', fontsize=12, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show(block=True)




### ---------- APPENDIX ---------- MULTI MODEL TRENDS PER AREA
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# SHRUBS file paths for each model's spreadsheet
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx'
]

# wet tundra
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425_FILTERED2.xlsx',
]

# lakes
file_paths = [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep_MEAN_yen_280325_FILTERED2.xlsx',
]

# shrubs
outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/scatterplots_allareas_models_cover_means_shrubs'

# wettundra save
outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_wettundra'

# lakes save
outdir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/scatterplots_models_cover_means_lakes'

modelnames = ['CNN 512-filters','ResNet50', 'VGG19', 'U-Net 256-filters', 'ViT']
count = 0
# Loop over each file to create a separate figure
for fp in file_paths:
    # Read the Excel file (adjust sheet name as necessary)
    df = pd.read_excel(fp, sheet_name='shrub_cover_statistics')

    # Clean up the 'File' column and extract numbers for sorting
    df['File'] = df['File'].str.replace('ms', 'site')
    df['site_num'] = df['File'].str.extract(r'(\d+)').astype(int)
    df['area_num'] = df['Area'].str.extract(r'(\d+)').astype(int)

    # Remove rows with non-finite slope or CI values
    df = df[np.isfinite(df['Slope Continuous']) &
            np.isfinite(df['CI Lower Continuous']) &
            np.isfinite(df['CI Upper Continuous'])]

    # Group by 'File' and 'Area' (if needed, depending on your aggregation requirements)
    df_grouped = df.groupby(['File', 'Area', 'site_num', 'area_num'], dropna=False).agg({
        'Slope Continuous': 'mean',
        'CI Lower Continuous': 'mean',
        'CI Upper Continuous': 'mean'
    }).reset_index()

    # Sort by site_num and area_num for logical x-axis ordering
    df_grouped = df_grouped.sort_values(by=['site_num', 'area_num'])

    # Create x positions for plotting (you can adjust gaps as needed)
    gap_intra = 0.1  # gap between areas within a site
    gap_inter = 0.3  # gap between sites
    current_x = 0
    labels = []

    for site in sorted(df_grouped['File'].unique(), key=lambda x: int(x.replace('site', ''))):
        sub = df_grouped[df_grouped['File'] == site].sort_values(by='area_num')
        n = len(sub)
        positions = current_x + np.arange(n) * gap_intra
        df_grouped.loc[sub.index, 'x'] = positions
        for pos, area in zip(positions, sub['Area']):
            labels.append(f"{site}-{area}")
        current_x = positions[-1] + gap_inter  # use positions[-1] instead of positions.iloc[-1]

    # Calculate error bars from slope and CI
    mean_slope = df_grouped['Slope Continuous']
    err_lower = mean_slope - df_grouped['CI Lower Continuous']
    err_upper = df_grouped['CI Upper Continuous'] - mean_slope
    yerr = [err_lower, err_upper]

    # Create a new figure for this model
    # fig, ax = plt.figure(figsize=(16, 6))
    plt.figure(figsize=(16, 6))
    plt.errorbar(df_grouped['x'], mean_slope, yerr=yerr,
                 fmt='o', capsize=6, color='blue',
                 markerfacecolor='white', markersize=10)
    plt.xticks(df_grouped['x'], df_grouped['File'] + '-' + df_grouped['Area'],
               rotation=45, fontsize=10, fontweight='bold')
    plt.xlabel('Site - Area', fontsize=16, fontweight='bold')
    # plt.ylabel('Trend Tall Shrub Cover', fontsize=16, fontweight='bold')
    # plt.ylabel('Trend Wet Tundra Cover', fontsize=16, fontweight='bold')
    plt.ylabel('Trend Surface Water \nBodies Cover', fontsize=16, fontweight='bold')

    # Extract model name from the file name (adjust based on your naming convention)
    # model_name = os.path.basename(fp).split('_')[5]
    model_name = modelnames[count]
    count += 1
    # plt.title(f'Per-Area Trends in Tall Shrub Cover (Continuous Predictions) for {model_name} \n with 95% CI intervals',
    #           fontsize=20, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.ylim(-0.6,0.6)
    plt.grid(False)
    plt.tight_layout()
    # Display or save the figure (you can change plt.show() to plt.savefig() if desired)
    # plt.show(block=True)

    ## SAVE FIGURE
    fn = f"TREND_PER_AREA_APPENDIX_{model_name}.jpg"
    plt.savefig(os.path.join(outdir, fn), dpi=200)
    plt.close(fig)








### boxplots Trends per site

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the 'File' column is formatted correctly for ordering.
df2['File'] = df2['File'].str.replace('ms', 'site')

# Create a numeric version for ordering if needed.
df2['site_num'] = df2['File'].str.extract(r'(\d+)').astype(int)

# Sort the DataFrame by site number (and optionally by another grouping variable)
df2 = df2.sort_values(by='site_num')

# Now plot boxplots of the continuous slopes per site.
plt.figure(figsize=(10, 6))
sns.boxplot(x='File', y='Slope Continuous', data=df2)
plt.xlabel('Site', fontsize=14, fontweight='bold')
plt.ylabel('Slope Continuous', fontsize=14, fontweight='bold')
plt.title('Distribution of Continuous Slopes per Site', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)













### STATISTICS AREAS , SENSORS, METHODS, CORRELATION WITH COVER FRACTIONS

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

# Directory containing Excel files
# SHRUB old file with more June months
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225'
# SHRUB new directory from march 2025 with all filtered
folder_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225"
# SHRUB CALIBRATED
folder_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_CALIBRATED_blocksize_200_edge_artifact_CI_bootstrap_extra_filtered_ms10a9_230325"


# wettundra model performance
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525'
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT'

# lakes model performance
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525'


dfs = []
for file in os.listdir(folder_path):
    if file.startswith("~$") or not file.endswith((".xlsx", ".xls")):
        continue
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path, sheet_name='shrub_cover_data')
    # Optionally, extract sensor information from the filename and add it as a column.
    # For example, if filenames are like "QB02_data.xlsx" or "WV03_data.xlsx":
    # sensor = file.split('_')[0]
    # df['Sensor'] = sensor
    dfs.append(df)
combined_df = pd.concat(dfs, ignore_index=True)



### Sensor vs cover
# nominal category with no inherent ranking, neither Pearson nor Spearman correlation makes sense

# “do mean cover fractions differ by sensor?” ANOVA/Kruskal–Wallis
metrics = ['cover_frac_binary', 'cover_frac_cont']
for metric in metrics:
    print(f"\n=== Testing {metric} ===")
    # 1) Normality (Shapiro-Wilk) per Method
    normality = {}
    for method, grp in combined_df.groupby('Sensor'):
        stat, p = stats.shapiro(grp[metric])
        normality[method] = p
        print(f"Shapiro test for {method}: p = {p:.3f}")
    # 2) Homogeneity of variances (Levene’s test)
    stat, p_levene = stats.levene(
        *[grp[metric].values for _, grp in combined_df.groupby('Sensor')])
    print(f"Levene’s test: p = {p_levene:.3f}")
    # 3) Choose test
    if all(p > 0.05 for p in normality.values()) and p_levene > 0.05:
        print("-> Use one‐way ANOVA")
        model = smf.ols(f'{metric} ~ C(Sensor)', data=combined_df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        print(anova)
    else:
        print("-> Use Kruskal–Wallis test")
        groups = [grp[metric] for _, grp in combined_df.groupby('Sensor')]
        H, p_kw = stats.kruskal(*groups)
        print(f"Kruskal–Wallis H-test: H = {H:.3f}, p = {p_kw:.3f}")


## Post hoc sensor:

# pip install scikit-posthocs
import scikit_posthocs as sp
import numpy as np

# Used Kruskal–Wallis for each metric, Dunn’s test below:
for metric in metrics:
    print(f"\n=== Post hoc for {metric} ===")
    # Prepare a list of arrays (one array per sensor), in the same order as in Kruskal–Wallis:
    groups = []
    labels = []
    for sensor, grp in combined_df.groupby("Sensor"):
        groups.append(grp[metric].values)
        labels.append(sensor)

    # Stack them into a single 1D array, and build a matching “group labels” array:
    data_all = np.concatenate(groups)
    group_labels = np.concatenate([
        np.full(len(g), labels[i]) for i, g in enumerate(groups)
    ])

    # Run Dunn’s test (two‐sided) with Bonferroni correction:
    posthoc = sp.posthoc_dunn(
        pd.DataFrame({metric: data_all, "Sensor": group_labels}),
        val_col=metric,
        group_col="Sensor",
        p_adjust="bonferroni"
    )

    print(posthoc.round(4))




### Cover vs File:
print(combined_df['File'].unique())
file_map = {'ms1':0, 'ms2':1, 'ms4':2, 'ms5':3, 'ms6':4, 'ms7':5, 'ms10':6, 'ms11':7, 'ms12':8, 'ms13':9, 'ms15':10, 'ms16':11}
combined_df['File_numeric'] = combined_df['File'].map(file_map)

# --> meaningless to compute a Pearson r using site codes as if they were a true numeric or ordinal variable
# pearson_corr, p_val = pearsonr(combined_df['File_numeric'], combined_df['cover_frac_binary'])
# print("Pearson correlation (File vs. cover_frac_binary):", pearson_corr, "p =", p_val)
model_file = smf.ols('cover_frac_binary ~ C(File)', data=combined_df).fit()
anova_file = sm.stats.anova_lm(model_file, typ=2)
print("ANOVA for File:\n", anova_file)
# --> meaningless to compute a Pearson r using site codes as if they were a true numeric or ordinal variable
# pearson_corr, p_val = pearsonr(combined_df['File_numeric'], combined_df['cover_frac_cont'])
# print("Pearson correlation (File vs. cover_frac_cont):", pearson_corr, "p =", p_val)
model_file = smf.ols('cover_frac_cont ~ C(File)', data=combined_df).fit()
anova_file = sm.stats.anova_lm(model_file, typ=2)
print("ANOVA for File:\n", anova_file)




### CORRELATION “Method” (or “File” or “Area”)
# No Pearson-correlation with method -- categorical (e.g. Otsu, Li, percentile, dynamic). ranking as 1–4 and then
# computing Pearson r vs. cover fraction treats as numeric scale.

# print(combined_df['Method'].unique())
# method_map = {'MEAN_p50':0, 'MEAN_p70':1, 'MEAN_p80':2, 'MEAN_otsu':3, 'MEAN_yen':4, 't0.5':5}
# combined_df['Method_numeric'] = combined_df['Method'].map(method_map)
# model_method = smf.ols('cover_frac_binary ~ C(Method)', data=combined_df).fit()
# anova_method = sm.stats.anova_lm(model_method, typ=2)
# print("ANOVA for Method:\n", anova_method)

# 1) Test normality in each Method group -- USELESS FOR CONTINUOUS COVER AS THIS DOES NOT USE a THRESHOLD
metrics = ['cover_frac_binary']
for metric in metrics:
    print(f"\n=== Testing {metric} ===")
    # 1) Normality (Shapiro-Wilk) per Method
    normality = {}
    for method, grp in combined_df.groupby('Method'):
        stat, p = stats.shapiro(grp[metric])
        normality[method] = p
        print(f"Shapiro test for {method}: p = {p:.3f}")
    # 2) Homogeneity of variances (Levene’s test)
    stat, p_levene = stats.levene(
        *[grp[metric].values for _, grp in combined_df.groupby('Method')])
    print(f"Levene’s test: p = {p_levene:.3f}")
    # 3) Choose test
    if all(p > 0.05 for p in normality.values()) and p_levene > 0.05:
        print("-> Use one‐way ANOVA")
        model = smf.ols(f'{metric} ~ C(Method)', data=combined_df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        print(anova)
    else:
        print("-> Use Kruskal–Wallis test")
        groups = [grp[metric] for _, grp in combined_df.groupby('Method')]
        H, p_kw = stats.kruskal(*groups)
        print(f"Kruskal–Wallis H-test: H = {H:.3f}, p = {p_kw:.3f}")
# Both the Shapiro–Wilk tests (p < 0.001 for every threshold method) and Levene’s test (p < 0.001) indicate that neither
# normality nor equal variances hold across the Method groups. Because those ANOVA assumptions are violated,
# a non-parametric Kruskal–Wallis test should be used instead of one-way ANOVA to compare cover fractions among threshold methods.






# # For Area:
# print(combined_df['Area'].unique())
# area_map = {'area1':0, 'area2':1, 'area3':2, 'area4':3, 'area5':4, 'area6':5, 'area7':6, 'area8':7, 'area9':8, 'area10':9, 'area11':10, 'area12':11, 'area13':12, 'area14':13}
# combined_df['Area_numeric'] = combined_df['Area'].map(area_map)
# pearson_corr, p_val = pearsonr(combined_df['Area_numeric'], combined_df['cover_frac_cont'])
# print("Pearson correlation (Area vs. cover_frac_cont):", pearson_corr, "p =", p_val)
#
# model_area = smf.ols('cover_frac_cont ~ C(Area)', data=combined_df).fit()
# anova_area = sm.stats.anova_lm(model_area, typ=2)
# print("ANOVA for Area:\n", anova_area)



### For Month:
print(combined_df['Month'].unique())
# month_map = {'6':0, '7':1, '8':2}
# combined_df['Month_numeric'] = combined_df['Month'].map(month_map)
# pearson_corr, p_val = pearsonr(combined_df['Month'], combined_df['cover_frac_cont'])
# print("Pearson correlation (Month vs. cover_frac_cont):", pearson_corr, "p =", p_val)
#
# model_area = smf.ols('cover_frac_cont ~ C(Month)', data=combined_df).fit()
# anova_area = sm.stats.anova_lm(model_area, typ=2)
# print("ANOVA for Area:\n", anova_area)

# 1) does cover tend to increase or decrease as summer progresses?
rho, p_spear = stats.spearmanr(combined_df['Month'], combined_df['cover_frac_binary'])
print(f"Spearman rho vs binary cover = {rho:.3f}, p = {p_spear:.3f}")

rho, p_spear = stats.spearmanr(combined_df['Month'], combined_df['cover_frac_cont'])
print(f"Spearman rho vs cont cover = {rho:.3f}, p = {p_spear:.3f}")


# 2) “do mean cover fractions differ by month?” ANOVA/Kruskal–Wallis

metrics = ['cover_frac_binary', 'cover_frac_cont']
for metric in metrics:
    print(f"\n=== Testing {metric} ===")
    # 1) Normality (Shapiro-Wilk) per Method
    normality = {}
    for method, grp in combined_df.groupby('Month'):
        stat, p = stats.shapiro(grp[metric])
        normality[method] = p
        print(f"Shapiro test for {method}: p = {p:.3f}")
    # 2) Homogeneity of variances (Levene’s test)
    stat, p_levene = stats.levene(
        *[grp[metric].values for _, grp in combined_df.groupby('Month')])
    print(f"Levene’s test: p = {p_levene:.3f}")
    # 3) Choose test
    if all(p > 0.05 for p in normality.values()) and p_levene > 0.05:
        print("-> Use one‐way ANOVA")
        model = smf.ols(f'{metric} ~ C(Month)', data=combined_df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        print(anova)
    else:
        print("-> Use Kruskal–Wallis test")
        groups = [grp[metric] for _, grp in combined_df.groupby('Month')]
        H, p_kw = stats.kruskal(*groups)
        print(f"Kruskal–Wallis H-test: H = {H:.3f}, p = {p_kw:.3f}")




### CORRELATION TRENDS



# Directory containing Excel file
folder_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225"

# wettundra model performance
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525'
# lakes model performance
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525'


dfs = []
for file in os.listdir(folder_path):
    if file.startswith("~$") or not file.endswith((".xlsx", ".xls")):
        continue
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path, sheet_name='shrub_cover_statistics')
    dfs.append(df)
combined_df2 = pd.concat(dfs, ignore_index=True)
combined_df2['Slope Continuous'].replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df2.dropna(subset=['Slope Continuous'], inplace=True)
combined_df2['Slope SE Continuous'].replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df2.dropna(subset=['Slope SE Continuous'], inplace=True)




### For n_years (to capture timeseries relationship):

# print(combined_df['File'].unique())
# file_map = {'ms1':0, 'ms2':1, 'ms4':2, 'ms5':3, 'ms6':4, 'ms7':5, 'ms10':6, 'ms11':7, 'ms12':8, 'ms13':9, 'ms15':10, 'ms16':11}
# combined_df['File_numeric'] = combined_df['File'].map(file_map)

# pearson_corr, p_val = pearsonr(combined_df2['n_years'], combined_df2['Slope Continuous'])
# print("Pearson correlation (File vs. Slope Continuous):", pearson_corr, "p =", p_val)

rho, p_spear = stats.spearmanr(combined_df2['n_years'], combined_df2['Slope Continuous'])
print(f"Spearman rank correlation: rho = {rho:.3f}, p = {p_spear:.3f}")


# model_file = smf.ols('Q("Slope Continuous") ~ C(n_years)', data=combined_df2).fit()
# anova_file = sm.stats.anova_lm(model_file, typ=2)
# print("ANOVA for File:\n", anova_file)
# ## Or
# X = sm.add_constant(combined_df2['n_years'])
# y = combined_df2['Slope Continuous']
# model = sm.OLS(y, X).fit()
# print(model.summary())
#
# # a) Scatter plot with regression line
# plt.figure()
# plt.scatter(combined_df2['n_years'], combined_df2['Slope Continuous'], alpha=0.6, label='Data')
# plt.plot(combined_df2['n_years'], model.predict(X), linewidth=2, label='Fit')
# plt.xlabel('Number of Time-series Scenes')
# plt.ylabel('Slope Continuous')
# plt.title('Time-series Count vs. Slope: Scatter with Fit')
# plt.legend()
# plt.show()
#
# # b) Residuals vs. Fitted values
# plt.figure()
# plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
# plt.axhline(0, linestyle='--')
# plt.xlabel('Fitted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs Fitted Values')
# plt.show()
#
# # c) QQ‐plot of residuals
# fig = sm.qqplot(model.resid, line='45')
# plt.title('QQ Plot of Residuals')
# plt.show()



# n_years vs Slope SE Continuous
dfs = []
for file in os.listdir(folder_path):
    if file.startswith("~$") or not file.endswith((".xlsx", ".xls")):
        continue
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path, sheet_name='shrub_cover_statistics')
    dfs.append(df)
combined_df2 = pd.concat(dfs, ignore_index=True)
combined_df2['Slope SE Continuous'].replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df2.dropna(subset=['Slope SE Continuous'], inplace=True)

rho, p_spear = stats.spearmanr(combined_df2['n_years'], combined_df2['Slope SE Continuous'])
print(f"Spearman rank correlation Slope SE Continuous: rho = {rho:.3f}, p = {p_spear:.3f}")





## OPTIONAL
# For n_years:

# combined_df3 = pd.concat(dfs, ignore_index=True)
# combined_df3['Slope SE Continuous'].replace([np.inf, -np.inf], np.nan, inplace=True)
# combined_df3.dropna(subset=['Slope SE Continuous'], inplace=True)
# pearson_corr, p_val = pearsonr(combined_df3['n_years'], combined_df3['Slope SE Continuous'])
# print("Pearson correlation (File vs. Slope SE Continuous):", pearson_corr, "p =", p_val)
#
# model_file = smf.ols('Q("Slope SE Continuous") ~ C(n_years)', data=combined_df3).fit()
# anova_file = sm.stats.anova_lm(model_file, typ=2)
# print("ANOVA for File:\n", anova_file)


# For n_years:
rho, p_spear = stats.spearmanr(combined_df2['n_years'], combined_df2['CI Upper Continuous'])
print(f"Spearman rank correlation Slope SE Continuous: rho = {rho:.3f}, p = {p_spear:.3f}")
# rho, p_val = pearsonr(combined_df2['n_years'], combined_df2['CI Upper Continuous'])
# print("Spearman correlation (File vs. CI Upper Continuous):", rho, "p =", p_val)


model_file = smf.ols('Q("CI Upper Continuous") ~ C(n_years)', data=combined_df2).fit()
anova_file = sm.stats.anova_lm(model_file, typ=2)
print("ANOVA for File:\n", anova_file)
