


# pip install statsmodels
# pip install openpyxl


## LOAD EXCEL
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr


def load_predictions(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import glob, os

from functionsdarko import (evaluate_thresholds_for_year2,
    evaluate_thresholds_for_year3_witherror,
    calculate_overall_significance_weighted
    )

# CNN p80 (bad correlation lakes - shrub. -0.06)
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# Resnet p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# VGG p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_yen_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# UNET p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# VIT p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_VITjune25/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_VIT/STATISTICS_shrub_cover_analysis_lakes_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_localCPU_v2.pthp_MEAN_p80_060625.xlsx', sheet_name='shrub_cover_statistics')


# ## SHRUB - load in all models for p80
# shrub_dir = (
#     "/Volumes/OWC Express 1M2/nasa_above/"
#     "predictions/saved_sheet/"
#     "saved_sheets_shrub_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9"
# )
# # match only the MEAN_p80 files
# pattern = os.path.join(shrub_dir, "STATISTICS_shrub_cover_analysis_shrub_*_MEAN_p80_*.xlsx")
# shrub_files_p80 = glob.glob(pattern)
# shrub_dfs = []
# for path in shrub_files_p80:
#     # read the one sheet you need
#     df = pd.read_excel(path, sheet_name="shrub_cover_statistics")
#     # extract a short model name from the filename
#     fname = os.path.basename(path)
#     # everything between the "shrub_" and the "_MEAN_p80"
#     model_name = fname.split("shrub_")[1].split("_MEAN_p80")[0]
#     # keep only the columns you want, rename slope
#     df = df[["File", "Area", "Slope Continuous"]].copy()
#     # df = df.rename(columns={"Slope Continuous": "Slope_Shrub"})
#     df["model"] = model_name
#     shrub_dfs.append(df)
# # concatenate your five MEAN_p80 runs into one DataFrame
# shrub_final_results_df = pd.concat(shrub_dfs, ignore_index=True)
#
#
#
# ## WET - load in all models for p80
# wet_dir = (
#     '/Volumes/OWC Express 1M2/nasa_above/predictions/saved_sheet/saved_sheets_wettundra_extra_filtered_ms10a9_040525'
# )
# # match only the MEAN_p80 files
# pattern = os.path.join(wet_dir, "STATISTICS_shrub_cover_analysis_wettundra_*_MEAN_p80_*.xlsx")
# wet_files_p80 = glob.glob(pattern)
# wet_dfs = []
# for path in wet_files_p80:
#     # read the one sheet you need
#     df = pd.read_excel(path, sheet_name="shrub_cover_statistics")
#     # extract a short model name from the filename
#     fname = os.path.basename(path)
#     # everything between the "shrub_" and the "_MEAN_p80"
#     model_name = fname.split("shrub_")[1].split("_MEAN_p80")[0]
#     # keep only the columns you want, rename slope
#     df = df[["File", "Area", "Slope Continuous"]].copy()
#     # df = df.rename(columns={"Slope Continuous": "Slope_Wet"})
#     df["model"] = model_name
#     wet_dfs.append(df)
# # concatenate your five MEAN_p80 runs into one DataFrame
# wettundra_final_results_df = pd.concat(wet_dfs, ignore_index=True)
#
#
#
#
# ## LAKES - load in all models for p80
# lake_dir = (
#     '/Volumes/OWC Express 1M2/nasa_above/predictions/saved_sheet/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525'
# )
# # match only the MEAN_p80 files
# pattern = os.path.join(lake_dir, "STATISTICS_shrub_cover_analysis_lakesrivers_*_MEAN_p80_*.xlsx")
# lakes_files_p80 = glob.glob(pattern)
# lake_dfs = []
# for path in lakes_files_p80:
#     # read the one sheet you need
#     df = pd.read_excel(path, sheet_name="shrub_cover_statistics")
#     # extract a short model name from the filename
#     fname = os.path.basename(path)
#     # everything between the "shrub_" and the "_MEAN_p80"
#     model_name = fname.split("shrub_")[1].split("_MEAN_p80")[0]
#     # keep only the columns you want, rename slope
#     df = df[["File", "Area", "Slope Continuous"]].copy()
#     # df = df.rename(columns={"Slope Continuous": "Slope_Wet"})
#     df["model"] = model_name
#     lake_dfs.append(df)
# # concatenate your five MEAN_p80 runs into one DataFrame
# lakesrivers_final_results_df = pd.concat(lake_dfs, ignore_index=True)








##  --- --- --- T-test --- --- ---
## GET TRENDS PER THRESHOLD PER MODEL

all_statistics_df = shrub_final_results_df
# all_statistics_df = wettundra_final_results_df
# all_statistics_df = lakesrivers_final_results_df

overall_significance_results = calculate_overall_significance_weighted(
    all_statistics_df,
    slope_column='Annual Slope',
    se_column='Annual Slope SE'
)
print(overall_significance_results)



### --> --> --> --> No JUNE  --> --> --> -->
overall_significance_results = calculate_overall_significance_weighted(
    all_statistics_df,
    slope_column='Annual Slope Ex June',
    se_column='Annual Slope SE Ex June'
)
print(overall_significance_results)





### -----> SEE -----> stats_extract_trends_excelfiles.py to extract TRENDS ---------------
## loop to print trends and significance

folder_path = "/Users/radakovicd1/Downloads/saved_sheets"
for file in sorted(os.listdir(folder_path)):
    if file.endswith('.xlsx'):
        all_statistics_df = pd.read_excel(os.path.join(folder_path, file), sheet_name='shrub_cover_statistics')
        print(file[32:-16])
        print(file[-16:])
        overall_significance_results = calculate_overall_significance_weighted(
            all_statistics_df,
            slope_column='Annual Slope',
            se_column='Annual Slope SE'
        )
        # print(overall_significance_results)
        print(overall_significance_results['Weighted Mean Slope'], 'significant: ',overall_significance_results['Significant'])
        ### --> --> --> --> No JUNE  --> --> --> -->
        overall_significance_results = calculate_overall_significance_weighted(
            all_statistics_df,
            slope_column='Annual Slope Ex June',
            se_column='Annual Slope SE Ex June'
        )
        # print(overall_significance_results)
        print(overall_significance_results['Weighted Mean Slope'], 'significant: ',overall_significance_results['Significant'])
        print('')


### -----> SEE -----> stats_extract_trends_excelfiles.py to extract TRENDS ---------------
## loop to print trends and significance 2025 with Binary and Continuous
folder_path = "/Users/radakovicd1/Downloads/saved_sheets"
for file in sorted(os.listdir(folder_path)):
    if file.endswith('.xlsx'):
        all_statistics_df = pd.read_excel(os.path.join(folder_path, file), sheet_name='shrub_cover_statistics')
        print(file[32:-16])
        print(file[-16:])

        overall_significance_results = calculate_overall_significance_weighted(
            all_statistics_df,
            slope_column='Slope Binary',
            se_column='Slope SE Binary'
        )
        # print(overall_significance_results)
        print('bin', f"{overall_significance_results['Weighted Mean Slope']:.4f}", 'significant: ',overall_significance_results['Significant'])

        overall_significance_results = calculate_overall_significance_weighted(
            all_statistics_df,
            slope_column='Slope Continuous',
            se_column='Slope SE Continuous'
        )
        # print(overall_significance_results)
        print('cont', f"{overall_significance_results['Weighted Mean Slope']:.4f}" , 'significant: ', overall_significance_results['Significant'])

        ### --> --> --> --> No JUNE BINARY --> --> --> -->
        overall_significance_results = calculate_overall_significance_weighted(
            all_statistics_df,
            slope_column='Slope Binary Ex June',
            se_column='Slope SE Binary Ex June'
        )
        # print(overall_significance_results)
        print('bin ex june', f"{overall_significance_results['Weighted Mean Slope']:.4f}", 'significant: ',overall_significance_results['Significant'])

        ### --> --> --> --> No JUNE Continuous --> --> --> -->
        overall_significance_results = calculate_overall_significance_weighted(
            all_statistics_df,
            slope_column='Slope Continuous Ex June',
            se_column='Slope SE Continuous Ex June'
        )
        # print(overall_significance_results)
        print('cont ex june', f"{overall_significance_results['Weighted Mean Slope']:.4f}", 'significant: ',
              overall_significance_results['Significant'])
        print('')





#  --- --- --- EXTRACT slopes --- --- ---
shrub_slopes = shrub_final_results_df['Slope Continuous']
# shrub_slopes = shrub_final_results_df['Annual Slope Ex June']
wet_tundra_slopes = wettundra_final_results_df['Slope Continuous']
# wet_tundra_slopes = wettundra_final_results_df['Annual Slope Ex June']
lakesrivers_slopes = lakesrivers_final_results_df['Slope Continuous']
# lakesrivers_slopes = lakesrivers_final_results_df['Annual Slope Ex June']


# ## For JUNE removal
# shrub_slopes = shrub_slopes[np.isfinite(shrub_slopes)]
# wet_tundra_slopes = wet_tundra_slopes[np.isfinite(wet_tundra_slopes)]




### --- --- --- Correlation --- --- ---
# pearson_corr, p_value = pearsonr(shrub_slopes, wet_tundra_slopes)
# print(f"Pearson Correlation: {pearson_corr}, p-value: {p_value}")

# shrub 256fil wet tundra 256f all months
# Pearson Correlation: 0.8701216229928778, p-value: 1.8361104951339195e-41

# shrub 256fil wet tundra 256f No JUNE
# Pearson Correlation: 0.8935211325943399, p-value: 9.509996889083163e-41



# Correlation
# pearson_corr, p_value = pearsonr(shrub_slopes, lakesrivers_slopes)
# print(f"Pearson Correlation: {pearson_corr}, p-value: {p_value}")


# Pearson Correlation: 0.6307893309759166, p-value: 6.759089768350598e-16


## UPDATE May 2025 WET TUNDRA
df_shrub = shrub_final_results_df.reset_index()[['File','Area','Slope Continuous']]
df_wet  = wettundra_final_results_df.reset_index()[['File','Area','Slope Continuous']]

# rename so we can distinguish them
df_shrub = df_shrub.rename(columns={'Slope Continuous':'Slope_Shrub'})
df_wet   = df_wet.rename(  columns={'Slope Continuous':'Slope_Wet'  })

# merge them on File+Area
df_both = pd.merge(df_shrub, df_wet, on=['File','Area'], how='inner')

# drop any rows where either slope is NaN
df_both = df_both.dropna(subset=['Slope_Shrub','Slope_Wet'])

# now grab the two clean arrays
x = df_both['Slope_Shrub'].values
y = df_both['Slope_Wet'].values

pearson_corr, p_value = pearsonr(x, y)
print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")



#
#
# I want to create a graph of the distance to the wet tundra and on the other axes the quantity of shrubs if possible?
# my shrub_map and wet_tundra_map are actually a list of smaller images the size of (400,400,1). Can I loop through them?



#### PLOT




### Wettundra with CI bands
pearson_corr, p_value = pearsonr(x, y)
print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")

X = df_both['Slope_Shrub'].values
Y = df_both['Slope_Wet'].values

# 1) fit an OLS model with intercept
X_sm = sm.add_constant(X)            # adds the constant term
ols = sm.OLS(Y, X_sm).fit()

# 2) get the predictions + confidence intervals
pred = ols.get_prediction(X_sm)
pred_df = pred.summary_frame(alpha=0.05)  # 95% CI

# grab everything into vectors and sort by X
order = np.argsort(X)
Xo = X[order]
yo = Y[order]
yhat = pred_df["mean"].values[order]
ci_lo = pred_df["mean_ci_lower"].values[order]
ci_hi = pred_df["mean_ci_upper"].values[order]

# # 3) plot
# plt.figure(figsize=(10,10))
# # plt.scatter(X, Y, alpha=0.6, s=100, label="data")
# plt.scatter(X, Y, alpha=0.6, s=100)

import matplotlib.ticker as mticker
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(X, Y, alpha=0.6, s=100)

# plot the regression line
plt.plot(Xo, yhat, color="red", lw=2,
         # label=f"y = {ols.params[1]:.3f}x + {ols.params[0]:.3f}\n"
         label=f"$R^2$ = {ols.rsquared:.2f}\n"
               f"r = {pearson_corr:.2f}, p = {p_value:.2e}")
# plot the CI band first
plt.fill_between(Xo, ci_lo, ci_hi,
                 color="red", alpha=0.3,
                 label="95% CI")
# # annotate stats
# plt.text(0.05, 0.95,
#          f"r = {pearson_corr:.2f}\np = {p_value:.2e}",
#          transform=plt.gca().transAxes,
#          va="top", ha="left",
#          fontsize=14, bbox=dict(fc="white", alpha=0.7))
# tell matplotlib to use at most 6 ticks on the x-axis
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))

# force tick labels to plain numbers (not offset/scientific)
ax.ticklabel_format(style='plain', axis='x')

plt.xlabel("Shrub Cover Trends", fontsize=25, fontweight="bold")
plt.ylabel("Wet Tundra Cover Trends", fontsize=25, fontweight="bold")
# plt.grid(linestyle="--", alpha=0.5)
plt.legend(fontsize=25, loc="best", labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
plt.xticks(fontsize=25, fontweight='bold')
plt.yticks(fontsize=25, fontweight='bold')
plt.tight_layout()
plt.show(block=True)


# Correlation WETTUNDRA  [OLD SCATTER PLOT]
# pearson_corr, p_value = pearsonr(shrub_slopes, wet_tundra_slopes)
# print(f"Pearson Correlation: {pearson_corr}, p-value: {p_value}")
# pearson_corr, p_value = pearsonr(x, y)
# print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")

# # X = shrub_slopes
# # Y = wet_tundra_slopes
#
# X = df_both['Slope_Shrub'].values
# Y = df_both['Slope_Wet'].values
#
# m, b = np.polyfit(X, Y, 1)  # linear fit
# predicted = m * X + b
# residuals = Y - predicted
# SSres = np.sum(residuals**2)
# SStot = np.sum((Y - np.mean(Y))**2)
# r_squared = 1 - (SSres / SStot)
#
# plt.figure(figsize=(10, 10))
# plt.scatter(X, Y, alpha=0.7, label='Data Points', s=200)
# plt.plot(X, predicted, color='red', label='Best Fit Line')
# plt.xlabel('Shrub Cover Trends', fontsize=25, fontweight='bold')
# plt.ylabel('Wet Tundra Cover Trends', fontsize=25, fontweight='bold')
# # plt.title('Relationship between Shrub and Wet Tundra Changes\n with Mean_p70 thresholding on 512fil model', fontsize=16, fontweight='bold')
# # plt.title('Relationship between Shrub and Wet Tundra Changes\n with Mean_Yen thresholding', fontsize=16, fontweight='bold')
# plt.title('Relationship between Shrub and Wet Tundra Changes \n', fontsize=20, fontweight='bold')
# # plt.xlim(-0.03, 0.03)
# # plt.ylim(-0.03, 0.04)
#
# plt.xticks(fontsize=20, fontweight='bold')
# plt.yticks(fontsize=20, fontweight='bold')
# # Add text for Pearson correlation and R²
# # Assuming you already computed pearson_corr and p_value
# # pearson_corr, p_value = pearsonr(X, Y)
# plt.text(0.05, 0.95,
#          f"r = {pearson_corr:.2f}\np = {p_value:.2e}\nR² = {r_squared:.2f}",
#          transform=plt.gca().transAxes,
#          fontsize=25, fontweight='bold',
#          verticalalignment='top',
#          bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
#
# # plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show(block=True)








# Correlation LAKES


## UPDATE May 2025 LAKES
df_shrub = shrub_final_results_df.reset_index()[['File','Area','Slope Continuous']]
df_lakes  = lakesrivers_final_results_df.reset_index()[['File','Area','Slope Continuous']]

# rename so we can distinguish them
df_shrub = df_shrub.rename(columns={'Slope Continuous':'Slope_Shrub'})
df_lakes   = df_lakes.rename(  columns={'Slope Continuous':'Slope_Lake'  })

# merge them on File+Area
df_both = pd.merge(df_shrub, df_lakes, on=['File','Area'], how='inner')

# drop any rows where either slope is NaN
df_both = df_both.dropna(subset=['Slope_Shrub','Slope_Lake'])

# now grab the two clean arrays
x = df_both['Slope_Shrub'].values
y2 = df_both['Slope_Lake'].values

pearson_corr, p_value = pearsonr(x, y2)
print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")



# # pearson_corr, p_value = pearsonr(shrub_slopes, lakesrivers_slopes)
# # print(f"Pearson Correlation: {pearson_corr}, p-value: {p_value}")
# pearson_corr, p_value = pearsonr(x, y)
# print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")
#
# # X = shrub_slopes
# # Y = lakesrivers_slopes
# X = df_both['Slope_Shrub'].values
# Y = df_both['Slope_Lake'].values
#
# m, b = np.polyfit(X, Y, 1)  # linear fit
# predicted = m * X + b
# residuals = Y - predicted
# SSres = np.sum(residuals**2)
# SStot = np.sum((Y - np.mean(Y))**2)
# r_squared = 1 - (SSres / SStot)
#
# plt.figure(figsize=(10, 6))
# plt.scatter(X, Y, alpha=0.7, label='Data Points')
# plt.plot(X, predicted, color='red', label='Best Fit Line')
# plt.xlabel('Shrub Cover Trends', fontsize=14, fontweight='bold')
# plt.ylabel('Surface Water Bodies Cover Trends', fontsize=14, fontweight='bold')
# # plt.title('Relationship between Shrub and Surface Water Bodies Changes\n with Mean_p70 thresholding', fontsize=16, fontweight='bold')
# # plt.title('Relationship between Shrub and Surface Water Bodies Changes\n with Mean_Yen thresholding', fontsize=16, fontweight='bold')
# plt.title('Relationship between Shrub and Surface Water Bodies Changes\n with Mean_Otsu thresholding', fontsize=16, fontweight='bold')
# plt.xlim(-0.03, 0.03)
# plt.ylim(-0.03, 0.04)
#
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# # Add text for Pearson correlation and R²
# # Assuming you already computed pearson_corr and p_value
# # pearson_corr, p_value = pearsonr(X, Y)
# plt.text(0.05, 0.95,
#          f"r = {pearson_corr:.2f}\np = {p_value:.2e}\nR² = {r_squared:.2f}",
#          transform=plt.gca().transAxes,
#          fontsize=12, fontweight='bold',
#          verticalalignment='top',
#          bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
#
# # plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show(block=True)




### LAKES with CI bands
pearson_corr, p_value = pearsonr(x, y2)
print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")

X = df_both['Slope_Shrub'].values
Y = df_both['Slope_Lake'].values

# 1) fit an OLS model with intercept
X_sm = sm.add_constant(X)            # adds the constant term
ols = sm.OLS(Y, X_sm).fit()

# 2) get the predictions + confidence intervals
pred = ols.get_prediction(X_sm)
pred_df = pred.summary_frame(alpha=0.05)  # 95% CI

# grab everything into vectors and sort by X
order = np.argsort(X)
Xo = X[order]
yo = Y[order]
yhat = pred_df["mean"].values[order]
ci_lo = pred_df["mean_ci_lower"].values[order]
ci_hi = pred_df["mean_ci_upper"].values[order]


# # 3) plot
# plt.figure(figsize=(10,10))
# # plt.scatter(X, Y, alpha=0.6, s=100, label="data")
# plt.scatter(X, Y, alpha=0.6, s=100)

import matplotlib.ticker as mticker
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(X, Y, alpha=0.6, s=100)

# plot the regression line
plt.plot(Xo, yhat, color="red", lw=2,
         # label=f"y = {ols.params[1]:.3f}x + {ols.params[0]:.3f}\n"
         label=f"$R^2$ = {ols.rsquared:.2f}\n"
               f"r = {pearson_corr:.2f}, p = {p_value:.2e}")
# plot the CI band first
plt.fill_between(Xo, ci_lo, ci_hi,
                 color="red", alpha=0.3,
                 label="95% CI")
# # annotate stats
# plt.text(0.05, 0.95,
#          f"r = {pearson_corr:.2f}\np = {p_value:.2e}",
#          transform=plt.gca().transAxes,
#          va="top", ha="left",
#          fontsize=14, bbox=dict(fc="white", alpha=0.7))
# tell matplotlib to use at most 6 ticks on the x-axis
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))

# force tick labels to plain numbers (not offset/scientific)
ax.ticklabel_format(style='plain', axis='x')

plt.xlabel("Shrub Cover Trends", fontsize=25, fontweight="bold")
plt.ylabel("Surface Water Bodies Cover Trends", fontsize=25, fontweight="bold")
# plt.grid(linestyle="--", alpha=0.5)
plt.legend(fontsize=25, loc="best", labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
plt.xticks(fontsize=25, fontweight='bold')
plt.yticks(fontsize=25, fontweight='bold')
plt.tight_layout()
plt.show(block=True)







# Correlation WET TUNDRA vs LAKES



## UPDATE May 2025 WET vs. LAKES
df_shrub = wettundra_final_results_df.reset_index()[['File','Area','Slope Continuous']]
df_lakes  = lakesrivers_final_results_df.reset_index()[['File','Area','Slope Continuous']]

# rename so we can distinguish them
df_shrub = df_shrub.rename(columns={'Slope Continuous':'Slope_Wet'})
df_lakes   = df_lakes.rename(  columns={'Slope Continuous':'Slope_Lake'  })

# merge them on File+Area
df_both = pd.merge(df_shrub, df_lakes, on=['File','Area'], how='inner')

# drop any rows where either slope is NaN
df_both = df_both.dropna(subset=['Slope_Wet','Slope_Lake'])

# now grab the two clean arrays
x2 = df_both['Slope_Wet'].values
y3 = df_both['Slope_Lake'].values

pearson_corr, p_value = pearsonr(x2, y3)
print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3e}")

# X = shrub_slopes
# Y = lakesrivers_slopes
X = df_both['Slope_Wet'].values
Y = df_both['Slope_Lake'].values

# 1) fit an OLS model with intercept
X_sm = sm.add_constant(X)            # adds the constant term
ols = sm.OLS(Y, X_sm).fit()

# 2) get the predictions + confidence intervals
pred = ols.get_prediction(X_sm)
pred_df = pred.summary_frame(alpha=0.05)  # 95% CI

# grab everything into vectors and sort by X
order = np.argsort(X)
Xo = X[order]
yo = Y[order]
yhat = pred_df["mean"].values[order]
ci_lo = pred_df["mean_ci_lower"].values[order]
ci_hi = pred_df["mean_ci_upper"].values[order]


import matplotlib.ticker as mticker
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(X, Y, alpha=0.6, s=100)

# plot the regression line
plt.plot(Xo, yhat, color="red", lw=2,
         # label=f"y = {ols.params[1]:.3f}x + {ols.params[0]:.3f}\n"
         label=f"$R^2$ = {ols.rsquared:.2f}\n"
               f"r = {pearson_corr:.2f}, p = {p_value:.2e}")
# plot the CI band first
plt.fill_between(Xo, ci_lo, ci_hi,
                 color="red", alpha=0.3,
                 label="95% CI")
# # annotate stats
# plt.text(0.05, 0.95,
#          f"r = {pearson_corr:.2f}\np = {p_value:.2e}",
#          transform=plt.gca().transAxes,
#          va="top", ha="left",
#          fontsize=14, bbox=dict(fc="white", alpha=0.7))
# tell matplotlib to use at most 6 ticks on the x-axis
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))

# force tick labels to plain numbers (not offset/scientific)
ax.ticklabel_format(style='plain', axis='x')

plt.xlabel("Wet Tundra Trends", fontsize=25, fontweight="bold")
plt.ylabel("Surface Water Bodies Cover Trends", fontsize=25, fontweight="bold")
# plt.grid(linestyle="--", alpha=0.5)
plt.legend(fontsize=25, loc="best", labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
plt.xticks(fontsize=25, fontweight='bold')
plt.yticks(fontsize=25, fontweight='bold')
plt.tight_layout()
plt.show(block=True)








# BOXPLOTS TRENDS PER SITE
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

# CNN p80 (bad correlation lakes - shrub. -0.06)
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# Resnet p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# VGG p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10_MEAN_yen_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# UNET p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525/STATISTICS_shrub_cover_analysis_lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24_MEAN_p80_280325_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')

# VIT p80
shrub_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9/STATISTICS_shrub_cover_analysis_shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask_MEAN_p80_220225_FILTERED2.xlsx', sheet_name='shrub_cover_statistics')
wettundra_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_VITjune25/STATISTICS_shrub_cover_analysis_wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth_MEAN_p80_060425.xlsx', sheet_name='shrub_cover_statistics')
lakesrivers_final_results_df = pd.read_excel('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_VIT/STATISTICS_shrub_cover_analysis_lakes_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_localCPU_v2.pthp_MEAN_p80_060625.xlsx', sheet_name='shrub_cover_statistics')

# 2) Extract the continuous‐slope column and the site identifier (“File”)
shrubs = shrub_final_results_df[["File","Slope Continuous"]].rename(columns={"Slope Continuous":"Slope"})
wet    = wettundra_final_results_df   [["File","Slope Continuous"]].rename(columns={"Slope Continuous":"Slope"})
lakes  = lakesrivers_final_results_df [["File","Slope Continuous"]].rename(columns={"Slope Continuous":"Slope"})

# 3) Choose which cover‐type to plot
df = shrubs   # or wet, or lakes

# 4) Group data by site
groups = df.groupby("File")["Slope"]

# 5) Prepare a list of arrays, one per site, and a list of site names

import re
def extract_num(s):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0
site_names = []
# site_names = sorted(site_names, key=lambda x: extract_num(x))
site_names = sorted(df['File'].unique(), key=lambda x: extract_num(x))


data = []
for site, series in groups:
    # site_names.append(site)
    data.append(series.values)



# 6) Draw the boxplot
fig, ax = plt.subplots(figsize=(12,6))
ax.boxplot(data, labels=site_names)

# 7) Tidy it up
ax.set_xlabel("Site (File)",   fontsize=14, fontweight="bold")
ax.set_ylabel("Cover‐trend (slope)", fontsize=14, fontweight="bold")
ax.set_title("Distribution of cover‐trends per site", fontsize=16)
ax.tick_params(axis="x", rotation=45, labelsize=12)
ax.tick_params(axis="y", labelsize=12)
plt.tight_layout()
plt.show(block=True)



import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

def extract_num(s):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else -1

# 1) Find all .xlsx files under a folder
base_dirs = {
    "shrub":"/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9",
    "wettundra":"/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT",
    "lakes":"/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_VIT",
}
# xlsx_paths = glob.glob(os.path.join(base_dir, "**", "STATISTICS_*_MEAN_p80_*.xlsx"), recursive=True)

# 2) Map each filename to a model label
model_map = {
    "cnn_model3":   "CNN",
    "RESNET50": "RESNET",
    "VGG19": "VGG",
    "UNET256":  "UNET",
    "VIT14":   "VIT",
}

df_list = []
for cover_type, basepath in base_dirs.items():
    pattern = os.path.join(basepath, "**", "STATISTICS_*_MEAN_p80_*.xlsx")
    for path in glob.glob(pattern, recursive=True):
        fn = os.path.basename(path)

        # 3) identify the model from filename
        for key, label in model_map.items():
            if key in fn:
                model_label = label
                break
        else:
            continue   # skip files that don’t match any model

        # 4) read the relevant sheet
        df = pd.read_excel(path,
                           sheet_name="shrub_cover_statistics",
                           usecols=["File","Area","Slope Continuous"])

        # 5) rename + tag
        df = df.rename(columns={"Slope Continuous":"Slope"})
        df["Model"] = model_label
        df["Cover"] = cover_type

        df_list.append(df)

big = pd.concat(df_list, ignore_index=True)


big["Slope_pct"] = big["Slope"] * 100.0

import matplotlib.pyplot as plt
import numpy as np



### ---- >>>>
df_shrub = big.query("Cover=='shrub'").dropna(subset=["Slope_pct"])
# df_shrub = big.query("Cover=='wettundra'").dropna(subset=["Slope_pct"])
# df_shrub = big.query("Cover=='lakes'").dropna(subset=["Slope_pct"])


# 2) Determine ordered list of sites (extracting numbers so “Site10” comes after “Site2”)
import re
def extract_num(s):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else -1

site_order = sorted(df_shrub["File"].unique(), key=extract_num)
model_order = ["CNN","RESNET","VGG","UNET","VIT"]

colors = {
    "CNN":   "#1f77b4",  # muted blue
    "RESNET":"#ff7f0e",  # orange
    "VGG":   "#2ca02c",  # green
    "UNET": "#d62728",  # red
    "VIT":   "#9467bd",  # purple
}

n_sites  = len(site_order)
n_models = len(model_order)
total_width = 0.8
width       = total_width / n_models

# 3) Compute horizontal offsets
offsets = np.linspace(-total_width/2 + width/2,
                      total_width/2 - width/2,
                      n_models)

# 4) Flatten data, positions, colors
all_data      = []
all_positions = []
all_colors    = []

for i, site in enumerate(site_order):
    for j, model in enumerate(model_order):
        arr = df_shrub[
            (df_shrub["File"] == site) &
            (df_shrub["Model"] == model)
        ]["Slope_pct"].values

        if arr.size == 0:
            continue

        all_data.append(arr)
        all_positions.append(i + offsets[j])
        all_colors.append(colors[model])

# 5) Draw once
fig, ax = plt.subplots(figsize=(12,6))
bp = ax.boxplot(all_data,
                positions=all_positions,
                widths=width,
                patch_artist=True,
                showfliers=False)

# 6) Color each box
for patch, c in zip(bp["boxes"], all_colors):
    patch.set_facecolor(c)
for median in bp["medians"]:
    median.set_color("black")

# 7) Final formatting
ax.set_xticks(np.arange(n_sites))
ax.set_xticklabels(site_order, rotation=45, ha="right")
ax.set_xlabel("Study Site", fontsize=20, fontweight="bold")
# ax.set_ylabel("Annual Shrub Cover \nTrend (%)", fontsize=20, fontweight="bold")
# ax.set_ylabel("Annual Wet Tundra Cover \nTrend (%)", fontsize=20, fontweight="bold")
ax.set_ylabel("Annual Surface-Water \nBodies Cover Trend (%)", fontsize=20, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold"); plt.yticks(fontsize=20, fontweight="bold")

# legend hack
for m in model_order:
    ax.plot([], [], color=colors[m], marker="s", linestyle="none", label=m)
ax.legend(loc="best", frameon=False, fontsize=20, title_fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)

plt.tight_layout()
plt.show(block=True)



## --- --- Stats boxplots per site
# ---- regardless of which model produced it

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp

df_shrub = big.query("Cover=='shrub'").dropna(subset=["Slope_pct"])
# df_shrub = big.query("Cover=='wettundra'").dropna(subset=["Slope_pct"])
# df_shrub = big.query("Cover=='lakes'").dropna(subset=["Slope_pct"])


# 1) Shapiro–Wilk normality test on Slope_pct by site
print("Shapiro–Wilk normality for Slope_pct by site:")
normality = {}
for site, grp in df_shrub.groupby("File")["Slope_pct"]:
    if len(grp) >= 3:
        stat, p = stats.shapiro(grp)
    else:
        p = 0.0  # force “non-normal” if too few samples
    normality[site] = p
    print(f"  {site:5s}  p = {p:.3f}")

# 2) Levene’s test for equal variances across sites
grouped = [grp.values for site, grp in df_shrub.groupby("File")["Slope_pct"] if len(grp) > 1]
if len(grouped) > 1 and all(len(g)>1 for g in grouped):
    _, p_levene = stats.levene(*grouped)
else:
    p_levene = 0.0
print(f"\nLevene’s test for homogeneity of variances: p = {p_levene:.3f}")

# 3) Choose ANOVA or Kruskal–Wallis
if all(p > 0.05 for p in normality.values()) and (p_levene > 0.05):
    print("\n→ Assumptions met; performing one-way ANOVA on Slope_pct by File")
    model = smf.ols("Slope_pct ~ C(File)", data=df_shrub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nOne-way ANOVA results:")
    print(anova_table)
else:
    print("\n→ Assumptions violated; performing Kruskal–Wallis test on Slope_pct by File")
    H, p_kw = stats.kruskal(*grouped)
    print(f"\nKruskal–Wallis H = {H:.3f}, p = {p_kw:.3f}")

# 4) Post-hoc Dunn’s test (Bonferroni-corrected)
print("\nPost-hoc Dunn’s test (Slope_pct by File):")
posthoc = sp.posthoc_dunn(
    df_shrub,
    val_col="Slope_pct",
    group_col="File",
    p_adjust="bonferroni"
)
print(posthoc.round(4))

















## LAKE DISTANCE
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt




shrub_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/predictions/shrub_cnn_model2_256fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_e39'
# shrub_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/predictions/shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_e8'
# shrub_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/predictions/shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep11'

wettundra_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/predictions/wettundra_cnn_model2_256fil_wettundra3_37im_RADnor_400px_PSHP_P1BS_v2'
# wettundra_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/predictions/wettundra_cnn_model3_512fil_wettundra3_37im_RADnor_400px_PSHP_P1BS_v4_ep14'

lakesrivers_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/predictions/lakesrivers_cnn_model2_256filRADnor_400px_PSHP_P1BS_v2_ep42'


## ms1a1
shrub_file = 'shrub_predictions_complete_ms1.pkl'
shrub_path = os.path.join(shrub_dir, shrub_file)
shrub_dict = load_predictions(shrub_path)
shrub_images = shrub_dict['area1']['QB02_2005']

wettundra_file = 'wettundra_predictions_complete_ms1.pkl'
wettundra_path = os.path.join(wettundra_dir, wettundra_file)
wet_tundra_dict = load_predictions(wettundra_path)
wet_tundra_images = wet_tundra_dict['area1']['QB02_2005']

lakesrivers_file = 'lakesrivers_predictions_complete_ms1.pkl'
lakesrivers_path = os.path.join(lakesrivers_dir, lakesrivers_file)
lakesrivers_dict = load_predictions(lakesrivers_path)
lakesrivers_images = lakesrivers_dict['area1']['QB02_2005']


del shrub_dict
del wet_tundra_dict
del lakesrivers_dict


## MEAN p70
shrub_predictions_dict_binary = {}
for year in shrub_images:
    _, shrub_predictions_dict_binary = evaluate_thresholds_for_year2(shrub_images, 'MEAN_p70')

wet_tundra_predictions_dict_binary = {}
for year in wet_tundra_images:
    _, wet_tundra_predictions_dict_binary = evaluate_thresholds_for_year2(wet_tundra_images, 'MEAN_p70')

lakesrivers_predictions_dict_binary = {}
for year in lakesrivers_images:
    _, lakesrivers_predictions_dict_binary = evaluate_thresholds_for_year2(lakesrivers_images, 'MEAN_p70')



## MEAN YEN
shrub_predictions_dict_binary = {}
for year in shrub_images:
    _, shrub_predictions_dict_binary = evaluate_thresholds_for_year2(shrub_images, 'MEAN_yen')

wet_tundra_predictions_dict_binary = {}
for year in wet_tundra_images:
    _, wet_tundra_predictions_dict_binary = evaluate_thresholds_for_year2(wet_tundra_images, 'MEAN_yen')

lakesrivers_predictions_dict_binary = {}
for year in lakesrivers_images:
    _, lakesrivers_predictions_dict_binary = evaluate_thresholds_for_year2(lakesrivers_images, 'MEAN_yen')



## MEAN OTSU
shrub_predictions_dict_binary = {}
for year in shrub_images:
    _, shrub_predictions_dict_binary = evaluate_thresholds_for_year2(shrub_images, 'MEAN_otsu')

wet_tundra_predictions_dict_binary = {}
for year in wet_tundra_images:
    _, wet_tundra_predictions_dict_binary = evaluate_thresholds_for_year2(wet_tundra_images, 'MEAN_otsu')

lakesrivers_predictions_dict_binary = {}
for year in lakesrivers_images:
    _, lakesrivers_predictions_dict_binary = evaluate_thresholds_for_year2(lakesrivers_images, 'MEAN_otsu')






# Storing pixel counts, instead of storing distances,
# Each shrub pixel has a fixed area: pixel_area_ha.
# The number of shrub pixels in each bin times pixel_area_ha gives area.


## Wet tundra

pixel_size_m = 0.5
pixel_area_m2 = pixel_size_m ** 2  # 0.25 m² per pixel
pixel_area_ha = pixel_area_m2 / 10_000  # Convert m² to hectares (1 ha = 10,000 m²)


distance_bins = [0, 5, 10, 20, 40, 80, 160]  # bins in meters

all_shrub_distances_m = []
for shrub_img, wet_img in zip(shrub_predictions_dict_binary, wet_tundra_predictions_dict_binary):
    shrub_img = np.squeeze(shrub_img)  # shape: (400,400)
    wet_img = np.squeeze(wet_img)  # shape: (400,400)

    # mask = True where NOT wet tundra
    mask = (wet_img == 0)
    distance_map = distance_transform_edt(mask)  # distance in pixels
    distance_map_m = distance_map * pixel_size_m  # convert to meters

    # Extract distances where shrub pixels == 1
    shrub_distances = distance_map_m[shrub_img == 1]
    all_shrub_distances_m.extend(shrub_distances)

all_shrub_distances_m = np.array(all_shrub_distances_m)

distance_bins = [0, 5, 10, 20, 40, 80, 160]  # example bins in meters
# Create a small DataFrame to use pd.cut easily
df_dist = pd.DataFrame({'distance': all_shrub_distances_m})

df_dist['distance_bin'] = pd.cut(df_dist['distance'], bins=distance_bins)

# Group by bin and extract the distances
box_data = []
labels = []
for i in range(len(distance_bins)-1):
    lower = distance_bins[i]
    upper = distance_bins[i+1]
    # Extract distances that fall into this bin
    bin_data = df_dist[(df_dist['distance'] > lower) & (df_dist['distance'] <= upper)]['distance']
    box_data.append(bin_data.values)
    labels.append(f"{lower}-{upper}m")


# Count how many shrub pixels are in each bin
area_per_bin = []
for i in range(len(distance_bins)-1):
    lower = distance_bins[i]
    upper = distance_bins[i+1]
    bin_count = np.sum((all_shrub_distances_m > lower) & (all_shrub_distances_m <= upper))
    # Convert count to area
    bin_area_ha = bin_count * pixel_area_ha
    area_per_bin.append(bin_area_ha)

plt.bar(labels, area_per_bin)
plt.xlabel('Distance Bin to Wet Tundra', fontsize=14, fontweight='bold')
plt.ylabel('Shrub Area (ha)', fontsize=14, fontweight='bold')
plt.title('Shrub Area by Distance to Wet Tundra' , fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.show()





## LAKES

pixel_size_m = 0.5
pixel_area_m2 = pixel_size_m ** 2  # 0.25 m² per pixel
pixel_area_ha = pixel_area_m2 / 10_000  # Convert m² to hectares (1 ha = 10,000 m²)


distance_bins = [0, 5, 10, 20, 40, 80, 160]  # example bins in meters



all_shrub_distances_m = []
for shrub_img, wet_img in zip(shrub_predictions_dict_binary, lakesrivers_predictions_dict_binary):
    shrub_img = np.squeeze(shrub_img)  # shape: (400,400)
    wet_img = np.squeeze(wet_img)  # shape: (400,400)

    # mask = True where NOT wet tundra
    mask = (wet_img == 0)
    distance_map = distance_transform_edt(mask)  # distance in pixels
    distance_map_m = distance_map * pixel_size_m  # convert to meters

    # Extract distances where shrub pixels == 1
    shrub_distances = distance_map_m[shrub_img == 1]
    all_shrub_distances_m.extend(shrub_distances)

all_shrub_distances_m = np.array(all_shrub_distances_m)

distance_bins = [0, 5, 10, 20, 40, 80, 160]  # example bins in meters
# Create a small DataFrame to use pd.cut easily
df_dist = pd.DataFrame({'distance': all_shrub_distances_m})

df_dist['distance_bin'] = pd.cut(df_dist['distance'], bins=distance_bins)

# Group by bin and extract the distances
box_data = []
labels = []
for i in range(len(distance_bins)-1):
    lower = distance_bins[i]
    upper = distance_bins[i+1]
    # Extract distances that fall into this bin
    bin_data = df_dist[(df_dist['distance'] > lower) & (df_dist['distance'] <= upper)]['distance']
    box_data.append(bin_data.values)
    labels.append(f"{lower}-{upper}m")


pixel_size_m = 0.5
pixel_area_m2 = pixel_size_m ** 2  # 0.25 m² per pixel
pixel_area_ha = pixel_area_m2 / 10_000  # Convert m² to hectares (1 ha = 10,000 m²)


distance_bins = [0, 5, 10, 20, 40, 80, 160]  # example bins in meters

# Count how many shrub pixels are in each bin
area_per_bin = []
for i in range(len(distance_bins)-1):
    lower = distance_bins[i]
    upper = distance_bins[i+1]
    bin_count = np.sum((all_shrub_distances_m > lower) & (all_shrub_distances_m <= upper))
    # Convert count to area
    bin_area_ha = bin_count * pixel_area_ha
    area_per_bin.append(bin_area_ha)

plt.bar(labels, area_per_bin)
plt.xlabel('Distance Bin to Surface Water Bodies', fontsize=14, fontweight='bold')
plt.ylabel('Shrub Area (ha)', fontsize=14, fontweight='bold')
plt.title('Shrub Area by Distance to Surface Water Bodies' , fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.show()










## Computes the Euclidean distance from each pixel to the nearest wet tundra pixel in pixel units

pixel_size_m = 0.5
pixel_area_m2 = pixel_size_m ** 2  # 0.25 m² per pixel
pixel_area_ha = pixel_area_m2 / 10_000  # Convert m² to hectares (1 ha = 10,000 m²)

all_shrub_distances_m = []

# shrub_predictions_dict_binary and wet_tundra_predictions_dict_binary
# are assumed to be lists of (400,400,1) binary arrays.
for shrub_img, wet_img in zip(shrub_predictions_dict_binary, wet_tundra_predictions_dict_binary):
    shrub_img = np.squeeze(shrub_img)  # shape: (400,400)
    wet_img = np.squeeze(wet_img)      # shape: (400,400)

    # mask = True where NOT wet tundra
    mask = (wet_img == 0)
    distance_map = distance_transform_edt(mask)  # distance in pixels
    distance_map_m = distance_map * pixel_size_m  # convert to meters

    # Extract distances where shrub pixels == 1
    shrub_distances = distance_map_m[shrub_img == 1]
    all_shrub_distances_m.extend(shrub_distances)

all_shrub_distances_m = np.array(all_shrub_distances_m)

# # Plot histogram of shrub distances in meters
# plt.figure(figsize=(10, 6))
# plt.hist(all_shrub_distances, bins=30, edgecolor='black', alpha=0.7)
# plt.xlabel('Distance to Wet Tundra (m)', fontsize=14)
# plt.ylabel('Number of Shrub Pixels', fontsize=14)
# plt.title('Distribution of Shrub Pixel Distances to Wet Tundra', fontsize=16)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
plt.hist(all_shrub_distances_m, bins=30,
         weights=np.full_like(all_shrub_distances_m, pixel_area_ha),
         edgecolor='black', alpha=0.7)
plt.xlabel('Distance to Wet Tundra (m)', fontsize=14)
plt.ylabel('Shrub Area (ha)', fontsize=14)
plt.title('Distribution of Shrub Area by Distance to Wet Tundra (in ha)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()








## LAKES DISTANCE
# Computes the Euclidean distance from each pixel to the nearest wet tundra pixel in pixel units

all_shrub_distances_m = []
for shrub_img, wet_img in zip(shrub_predictions_dict_binary, lakesrivers_predictions_dict_binary):
    shrub_img = np.squeeze(shrub_img)  # shape: (400,400)
    wet_img = np.squeeze(wet_img)  # shape: (400,400)

    # mask = True where NOT wet tundra
    mask = (wet_img == 0)
    distance_map = distance_transform_edt(mask)  # distance in pixels
    distance_map_m = distance_map * pixel_size_m  # convert to meters

    # Extract distances where shrub pixels == 1
    shrub_distances = distance_map_m[shrub_img == 1]
    all_shrub_distances_m.extend(shrub_distances)

all_shrub_distances_m = np.array(all_shrub_distances_m)

# # Plot histogram of shrub distances in meters
# plt.figure(figsize=(10, 6))
# plt.hist(all_shrub_distances, bins=30, edgecolor='black', alpha=0.7)
# plt.xlabel('Distance to Surface Water Bodies (m)', fontsize=14)
# plt.ylabel('Number of Shrub Pixels', fontsize=14)
# plt.title('Distribution of Shrub Pixel Distances to Surface Water Bodies', fontsize=16)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(10, 6))
plt.hist(all_shrub_distances_m, bins=20,
         weights=np.full_like(all_shrub_distances_m, pixel_area_ha),
         edgecolor='black', alpha=0.7)
plt.xlabel('Distance to Surface Water Bodies (m)', fontsize=14)
plt.ylabel('Shrub Area (ha)', fontsize=14)
plt.title('Distribution of Shrub Area by Distance to Surface Water Bodies for area ms1a1 (in ha)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()








## Relationship Shrub Cover Trend and Wet tundra cover trend
plt.figure(figsize=(10, 6))
plt.scatter(shrub_slopes, wet_tundra_slopes, alpha=0.7)
# Add a best-fit line using linear regression
m, b = np.polyfit(shrub_slopes, wet_tundra_slopes, 1)
plt.plot(shrub_slopes, m*shrub_slopes+b, color='red')
plt.xlabel('Shrub Cover Trends', fontsize=14, fontweight='bold')
plt.ylabel('Wet Tundra Cover Trends', fontsize=14, fontweight='bold')
plt.title('Relationship between Shrub and Wet Tundra Changes' , fontsize=16 , fontweight='bold')
plt.show()





## Relationship Shrub Cover Trend and Wet tundra cover trend
plt.figure(figsize=(10, 6))
plt.scatter(shrub_slopes, lakesrivers_slopes, alpha=0.7)
# Add a best-fit line using linear regression
m, b = np.polyfit(shrub_slopes, wet_tundra_slopes, 1)
plt.plot(shrub_slopes, m*shrub_slopes+b, color='red')
plt.xlabel('Shrub Cover Trends', fontsize=14, fontweight='bold')
plt.ylabel('Surface Water Bodies Trends', fontsize=14, fontweight='bold')
plt.title('Relationship between Shrub and Surface Water Bodies Changes' , fontsize=16 , fontweight='bold')
plt.show()













