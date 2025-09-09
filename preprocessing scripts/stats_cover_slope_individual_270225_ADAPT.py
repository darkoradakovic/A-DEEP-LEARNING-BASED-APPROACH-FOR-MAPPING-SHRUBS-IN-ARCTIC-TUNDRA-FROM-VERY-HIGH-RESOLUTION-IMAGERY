# Load predictions

# pip install statsmodels
# pip install openpyxl

import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from skimage.filters import threshold_yen
# from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li

from functionsdarko import (evaluate_thresholds_for_year2,
    evaluate_thresholds_for_year3_witherror, multisite_configurations,
    calculate_statistics_and_outliers_shrub_cover2dec24, calculate_changes_and_correlation,
    calculate_overall_significance_weighted, combine_tiles_to_large_image_withoutmeta,
    block_bootstrap_shrub_cover_continuous, block_bootstrap_shrub_cover_binary,
    calculate_statistics_and_outliers_shrub_cover6feb2025, combine_tiles_to_large_image_predictionsoneyear2,
    calculate_changes_and_correlation_v2, multisite_configurations_ADAPT, calculate_statistics_and_outliers_shrub_cover27feb2025
    )


def load_predictions(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)



def calculate_changes_and_correlation_v2(shrub_predictions, methods, multisite_key, multisite_configurations, meta_dict):
    """
    Calculate changes in shrub cover and correlation, incorporating multisite configurations for additional metadata.
    """
    all_shrub_data = []

    if multisite_key not in multisite_configurations:
        raise ValueError(f"Multisite configuration for {multisite_key} not found.")

    multisite_config = multisite_configurations[multisite_key]
    filenames = multisite_config['filenames']

    for method in methods:
        for area in shrub_predictions:
            shrub_predictions_area = shrub_predictions[area]

            # Initialize dictionaries to store data per year
            metrics_by_year = {}
            for year in shrub_predictions_area:

                if area not in meta_dict or year not in meta_dict[area] or not meta_dict[area][year]:
                    print(f"Skipping year {year} for area {area} because metadata is empty.")
                    continue

                # Get shrub data and per-image fractions
                (shrub_cover_m2, total_area_shrubmap_m2, shrub_cover_frac, shrub_predictions_binary,
                 pred_mean_shrub, pred_std_shrub, threshold_mean_shrub, threshold_std_shrub,
                 shrub_cover_per_image_m2, shrub_cover_fraction_per_image) = evaluate_thresholds_for_year3_witherror(
                    shrub_predictions_area[year], method)

                # grid_shape = (10, 10)
                # combined_image = combine_tiles_to_large_image_withoutmeta(shrub_predictions_area[year], grid_shape, ddtype=np.float32)
                # combined_binaryimage = combine_tiles_to_large_image_withoutmeta(shrub_predictions_binary, grid_shape, ddtype=np.float32)

                combined_image, _ = combine_tiles_to_large_image_predictionsoneyear2(shrub_predictions_area[year], meta_dict[area][year])
                print('year', year)
                print('area', area)
                combined_binaryimage, _ = combine_tiles_to_large_image_predictionsoneyear2(shrub_predictions_binary, meta_dict[area][year])

                mean_cov_cont, std_cov_cont, ci_low_cont, ci_up_cont = block_bootstrap_shrub_cover_continuous(combined_image, block_size=(200, 200), n_boot=10000)
                # mean_cov_b, std_cov_b, ci_low_b, ci_up_b = block_bootstrap_shrub_cover_binary(combined_predictionimage_MEAN_p70)
                mean_cov_b, std_cov_b, ci_low_b, ci_up_b = block_bootstrap_shrub_cover_binary(combined_binaryimage, block_size=(200, 200), n_boot=10000)
                # The standard deviation of these bootstrap is an estimate of the standard error of the
                # mean shrub cover estimate. The standard error (SE) of a statistic (like the mean) is defined as the standard
                # deviation of its sampling distribution. The use of 1.96 assumes that the distribution of the bootstrap estimates
                # is roughly normal

                # Save the comput*ed metrics for this year.
                metrics_by_year[year] = {
                    'cover_frac_binary': mean_cov_b,  # from binary predictions
                    'cover_frac_cont': mean_cov_cont,  # from continuous predictions
                    'std_cov_cont': std_cov_cont,
                    'ci_low_cont': ci_low_cont,
                    'ci_up_cont': ci_up_cont,
                    'std_cov_b': std_cov_b,
                    'ci_low_b': ci_low_b,
                    'ci_up_b': ci_up_b,
                    'cover_m2': shrub_cover_m2,
                    'total_area_shrubmap_m2': total_area_shrubmap_m2,
                }

            sorted_years = sorted(metrics_by_year.keys())
            for year in sorted_years:
                # For metadata matching: get the scene_id from filenames that starts with the year.
                scene_id = next((f for f in filenames if f.startswith(year)), None)
                if not scene_id:
                    raise ValueError(f"Scene ID for year {year} not found in filenames.")
                # Extract date parts (assuming a specific filename format).
                date_part = scene_id.split("_")[1]
                month = int(date_part[4:6])
                day = int(date_part[6:8])

                # Retrieve the per-year metrics.
                m = metrics_by_year[year]

                # Save the final results for this year.
                all_shrub_data.append({
                    'File': multisite_key,
                    'Area': area,
                    'Year': int(year.split('_')[1]),
                    'Month': month,
                    'Day': day,
                    'Method': method,
                    'cover_frac_binary': m['cover_frac_binary'],
                    'cover_frac_cont': m['cover_frac_cont'],
                    'std_cov_cont': m['std_cov_cont'],
                    'ci_low_cont': m['ci_low_cont'],
                    'ci_up_cont': m['ci_up_cont'],
                    'std_cov_b': m['std_cov_b'],
                    'ci_low_b': m['ci_low_b'],
                    'ci_up_b': m['ci_up_b'],
                    'cover_m2': m['cover_m2'],
                    'total_area_m2': m['total_area_shrubmap_m2'],
                    'Sensor': year.split('_')[0],
                    'scene_id': scene_id
                })
        return all_shrub_data




# Loop through directories and process files
# Loop through directories and process files
def process_directories(shrub_dir, methods, multisite_configurations):
    all_shrub_cover_data = []
    all_statistics = []

    shrub_files = sorted([f for f in os.listdir(shrub_dir) if f.endswith('.pkl')])

    for shrub_file in shrub_files:  # shrub_file = 'shrub_predictions_complete_ms11.pkl'

        if 'meta' not in shrub_file:
            shrub_path = os.path.join(shrub_dir, shrub_file)

            meta_path = os.path.join(shrub_dir, shrub_file.replace('predictions','meta'))

            multisite_key = shrub_file.split('_ms')[-1].split('.')[0]
            multisite_key = f"ms{multisite_key}"
            print(multisite_key)

            shrub_predictions = load_predictions(shrub_path)
            meta_dict = load_predictions(meta_path)
            all_shrub_data = calculate_changes_and_correlation_v2(shrub_predictions, methods, multisite_key, multisite_configurations, meta_dict)
            df = pd.DataFrame(all_shrub_data)

            # statistics_and_outliers = calculate_statistics_and_outliers_overlap_cover(df)
            # statistics_and_outliers = calculate_statistics_and_outliers_shrub_cover6feb2025(df, multisite_key)
            statistics_and_outliers = calculate_statistics_and_outliers_shrub_cover27feb2025(df, multisite_key, shrub_predictions, meta_dict)
            # for result in statistics_and_outliers:
            #     result['File'] = multisite_key
            all_statistics.extend(statistics_and_outliers)

            all_shrub_cover_data.append(df)

    # Combine results
    all_shrub_cover_df = pd.concat(all_shrub_cover_data, ignore_index=True)
    all_statistics_df = pd.DataFrame(all_statistics)

    return all_shrub_cover_df, all_statistics_df



shrub_dir_base_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/'

### >>>>>>>> CHANGE MANUALLY >>>>>>>
# path = 'shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask'
# path = 'shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask'
# path = 'shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask'
# path = 'shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask'
# path = 'shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask'

### >>>>>>>> CHANGE MANUALLY >>>>>>>
### WET TUNDRA
# path = 'wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18'
# path = 'wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21'
# path = 'wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21'
# path = 'wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4'
# path = 'wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48hours.pth'

### LAKESRIVERS
# path = 'lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24'
path = 'lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22'
# path = 'lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10'
# path = 'lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24'
# path = 'lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep'

date = '280325'

shrub_dir = os.path.join(shrub_dir_base_path,path)
# output_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/saved_sheets2/'
# output_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/saved_sheets_wettundra/'
output_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/saved_sheets_lakesrivers/'
os.makedirs(output_dir, exist_ok=True)

## test
# shrub_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/test'




methods = ['MEAN_p50']
# # shrub
final_results_df, results_slope_overlap = process_directories(shrub_dir, methods, multisite_configurations_ADAPT)
significance_results_bin = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Binary',se_column='Slope SE Binary')
significance_results_cont = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Continuous',se_column='Slope SE Continuous')
significance_results_bin['name'] = 'june-incl binary'
significance_results_cont['name'] = 'june-incl continuous'
df_significance = pd.DataFrame([significance_results_bin, significance_results_cont])
savename_stats = 'STATISTICS_shrub_cover_analysis_' + shrub_dir.split('/')[-1] + '_' + methods[0] + '_'  + date + '.xlsx'
savename_stats2 = os.path.join(output_dir,savename_stats)
with pd.ExcelWriter(savename_stats2) as writer:
    final_results_df.to_excel(writer, sheet_name='shrub_cover_data', index=False)
    results_slope_overlap.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
    df_significance.to_excel(writer, sheet_name='trends', index=False)


methods = ['MEAN_p70']
# # shrub
final_results_df, results_slope_overlap = process_directories(shrub_dir, methods, multisite_configurations_ADAPT)
significance_results_bin = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Binary',se_column='Slope SE Binary')
significance_results_cont = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Continuous',se_column='Slope SE Continuous')
significance_results_bin['name'] = 'june-incl binary'
significance_results_cont['name'] = 'june-incl continuous'
df_significance = pd.DataFrame([significance_results_bin, significance_results_cont])
savename_stats = 'STATISTICS_shrub_cover_analysis_' + shrub_dir.split('/')[-1] + '_' + methods[0] + '_'  + date + '.xlsx'
savename_stats2 = os.path.join(output_dir,savename_stats)
with pd.ExcelWriter(savename_stats2) as writer:
    final_results_df.to_excel(writer, sheet_name='shrub_cover_data', index=False)
    results_slope_overlap.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
    df_significance.to_excel(writer, sheet_name='trends', index=False)


methods = ['MEAN_p80']
# # shrub
final_results_df, results_slope_overlap = process_directories(shrub_dir, methods, multisite_configurations_ADAPT)
significance_results_bin = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Binary',se_column='Slope SE Binary')
significance_results_cont = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Continuous',se_column='Slope SE Continuous')
significance_results_bin['name'] = 'june-incl binary'
significance_results_cont['name'] = 'june-incl continuous'
df_significance = pd.DataFrame([significance_results_bin, significance_results_cont])
savename_stats = 'STATISTICS_shrub_cover_analysis_' + shrub_dir.split('/')[-1] + '_' + methods[0] + '_'  + date + '.xlsx'
savename_stats2 = os.path.join(output_dir,savename_stats)
with pd.ExcelWriter(savename_stats2) as writer:
    final_results_df.to_excel(writer, sheet_name='shrub_cover_data', index=False)
    results_slope_overlap.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
    df_significance.to_excel(writer, sheet_name='trends', index=False)


methods = ['MEAN_otsu']
# # shrub
final_results_df, results_slope_overlap = process_directories(shrub_dir, methods, multisite_configurations_ADAPT)
significance_results_bin = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Binary',se_column='Slope SE Binary')
significance_results_cont = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Continuous',se_column='Slope SE Continuous')
significance_results_bin['name'] = 'june-incl binary'
significance_results_cont['name'] = 'june-incl continuous'
df_significance = pd.DataFrame([significance_results_bin, significance_results_cont])
savename_stats = 'STATISTICS_shrub_cover_analysis_' + shrub_dir.split('/')[-1] + '_' + methods[0] + '_'  + date + '.xlsx'
savename_stats2 = os.path.join(output_dir,savename_stats)
with pd.ExcelWriter(savename_stats2) as writer:
    final_results_df.to_excel(writer, sheet_name='shrub_cover_data', index=False)
    results_slope_overlap.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
    df_significance.to_excel(writer, sheet_name='trends', index=False)


methods = ['MEAN_yen']
# # shrub
final_results_df, results_slope_overlap = process_directories(shrub_dir, methods, multisite_configurations_ADAPT)
significance_results_bin = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Binary',se_column='Slope SE Binary')
significance_results_cont = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Continuous',se_column='Slope SE Continuous')
significance_results_bin['name'] = 'june-incl binary'
significance_results_cont['name'] = 'june-incl continuous'
df_significance = pd.DataFrame([significance_results_bin, significance_results_cont])
savename_stats = 'STATISTICS_shrub_cover_analysis_' + shrub_dir.split('/')[-1] + '_' + methods[0] + '_'  + date + '.xlsx'
savename_stats2 = os.path.join(output_dir,savename_stats)
with pd.ExcelWriter(savename_stats2) as writer:
    final_results_df.to_excel(writer, sheet_name='shrub_cover_data', index=False)
    results_slope_overlap.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
    df_significance.to_excel(writer, sheet_name='trends', index=False)





methods = ['t0.5']
# # shrub
final_results_df, results_slope_overlap = process_directories(shrub_dir, methods, multisite_configurations_ADAPT)
significance_results_bin = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Binary',se_column='Slope SE Binary')
significance_results_cont = calculate_overall_significance_weighted(results_slope_overlap, slope_column='Slope Continuous',se_column='Slope SE Continuous')
significance_results_bin['name'] = 'june-incl binary'
significance_results_cont['name'] = 'june-incl continuous'
df_significance = pd.DataFrame([significance_results_bin, significance_results_cont])
savename_stats = 'STATISTICS_shrub_cover_analysis_' + shrub_dir.split('/')[-1] + '_' + methods[0] + '_'  + date + '.xlsx'
savename_stats2 = os.path.join(output_dir,savename_stats)
with pd.ExcelWriter(savename_stats2) as writer:
    final_results_df.to_excel(writer, sheet_name='shrub_cover_data', index=False)
    results_slope_overlap.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
    df_significance.to_excel(writer, sheet_name='trends', index=False)





