


# Darko Radakovic
# Montclair State University
# 23-03-25

### USE AFTER CREATION of excel files with stats_cover_slope_individual_270225_ADAPT.py (or newer code)

# This code will filter further into detail, through all SAVED SHEETS from a directory.

# Next, use stats_extract_trends_excelfiles.py for better copy paste of TRENDS to excel spreadsheet

# Saves spreadsheet output)


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
    calculate_changes_and_correlation_v2, calculate_statistics2
    )


# allowed
site_years = {
    'ms1': [2005, 2013, 2017, 2020],
    'ms2': [2002, 2014],
    'ms4': [2002, 2010, 2013, 2014, 2018],
    'ms5': [2003, 2010, 2022],
    'ms6': [2009, 2011, 2013, 2016, 2017],
    'ms7': [2002, 2014, 2020],
    'ms10': [2005, 2014, 2017],
    'ms11': [2002, 2020],
    'ms12': [2002, 2012],
    'ms13': [2002, 2008],
    'ms15': [2002],
    'ms16': [2004]
}

# not allowed
site_areas = {
    'ms4': { 2014:['area4','area5']},   # cloudy and outliers for VIT model
    'ms1': { 2017:['area5'],2020:['area5']}   # cloudy and outliers for VIT model
}


## SHRUB CALIBRATION
calibration_slopes = {
    'cnn': 0.86,
    'RESNET50': 0.94,
    'VIT14': 0.79,
    'VGG19': 0.83,
    'UNET256fil': 1.02
}


## WETTUNDRA CALIBRATION
calibration_slopes = {
    'cnn': 1.0,
    'RESNET50': 1.0,
    'VIT14': 1.0,
    'VGG19': 1.0,
    'UNET256fil': 1.0
}


def calculate_statistics_and_outliers_shrub_cover10feb2025(df, multisite_key, site_years, site_areas):
    """
    Calculate changes and outlier statistics for shrub cover over years using both the
    binary-based cover and the continuous prediction-based cover, but only for those years
    specified in the site_years dictionary for the given multisite_key.

    This function expects that the DataFrame df contains at least the following columns:
      - 'Area'
      - 'Method'
      - 'Year'            (numeric year, e.g. 2010)
      - 'cover_frac_binary'  (shrub cover fraction computed from binary maps)
      - 'cover_frac_cont'    (shrub cover fraction computed from continuous predictions)
      - 'Month'           (numeric month)

    It computes, separately for the binary and continuous cover values:
      - Mean, standard deviation,
      - Regression statistics via calculate_statistics2 (Slope, Slope SE, R², Confidence Interval, RMSE),
      - Outlier detection based on mean ± 1 SD.
    It also computes similar statistics for a subset of the data where Month != 6 (i.e. non-June data).

    Parameters:
      df (DataFrame): The input DataFrame with shrub cover data.
      multisite_key: The key (e.g. integer 1, 2, etc.) corresponding to the current site.
      site_years (dict): Dictionary with multisite keys and lists of allowed years.

    Returns:
      A list of dictionaries (one per Area/Method combination) with the computed statistics.
    """

    # First, filter the DataFrame to include only rows where 'Year' is in the allowed years.
    if multisite_key in site_years:
        allowed_years = site_years[multisite_key]
        df = df[df['Year'].isin(allowed_years)]

        df = df[df['File'] == multisite_key]
    else:
        print(f"Warning: multisite_key {multisite_key} not found in site_years; using full dataset.")

    results = []

    # Loop over each Area and Method combination.
    for area in df['Area'].unique():
        for method in df['Method'].unique():
            subset = df[(df['Area'] == area) & (df['Method'] == method)]

            # Remove rows whose area is disallowed for this multisite_key.
            if multisite_key in site_areas:
                disallowed = site_areas[multisite_key]
                # Only remove rows if the row's Year is a key in disallowed and the area is in the list.
                subset = subset[~subset.apply(
                    lambda row: (row['Year'] in disallowed) and (row['Area'] in disallowed[row['Year']]),
                    axis=1
                )]
            else:
                print(f"Warning: multisite_key {multisite_key, area} not found in site_areas; using full dataset.")

            # Ensure that we work only on valid (non-zero, non-NaN) values.
            valid_binary = subset[subset['cover_frac_binary'] > 0].dropna(subset=['cover_frac_binary'])
            valid_cont = subset[subset['cover_frac_cont'] > 0].dropna(subset=['cover_frac_cont'])

            # Compute statistics for binary-based shrub cover.
            if len(valid_binary) > 1:
                years_bin = np.array(valid_binary['Year']).reshape(-1, 1)
                cover_bin = valid_binary['cover_frac_binary'].values
                mean_cover_bin = np.mean(cover_bin)
                std_cover_bin = np.std(cover_bin, ddof=1)
                stats_bin = calculate_statistics2(years_bin.flatten(), cover_bin)
                slope_bin = stats_bin.get('Slope', None)
                r2_bin = stats_bin.get('R²', None)
                slope_se_bin = stats_bin.get('Slope SE', None)
                ci_bin = stats_bin.get('Confidence Interval', (None, None))
                rmse_bin = stats_bin.get('RMSE', None)
            else:
                mean_cover_bin = std_cover_bin = slope_bin = r2_bin = slope_se_bin = rmse_bin = np.nan
                ci_bin = (np.nan, np.nan)
                cover_bin = np.array([])

            # Compute statistics for continuous-based shrub cover.
            if len(valid_cont) > 1:
                years_cont = np.array(valid_cont['Year']).reshape(-1, 1)

                cover_cont = valid_cont['cover_frac_cont'].values
                mean_cover_cont = np.mean(cover_cont)
                std_cover_cont = np.std(cover_cont, ddof=1)
                stats_cont = calculate_statistics2(years_cont.flatten(), cover_cont)
                slope_cont = stats_cont.get('Slope', None)
                r2_cont = stats_cont.get('R²', None)
                slope_se_cont = stats_cont.get('Slope SE', None)
                ci_cont = stats_cont.get('Confidence Interval', (None, None))
                rmse_cont = stats_cont.get('RMSE', None)
            else:
                mean_cover_cont = std_cover_cont = slope_cont = r2_cont = slope_se_cont = rmse_cont = np.nan
                ci_cont = (np.nan, np.nan)
                cover_cont = np.array([])

            # Outlier detection for binary cover:
            if not np.isnan(mean_cover_bin) and not np.isnan(std_cover_bin):
                outliers_bin = [x for x in cover_bin if
                                x < mean_cover_bin - std_cover_bin or x > mean_cover_bin + std_cover_bin]
            else:
                outliers_bin = []

            # Outlier detection for continuous cover:
            if not np.isnan(mean_cover_cont) and not np.isnan(std_cover_cont):
                outliers_cont = [x for x in cover_cont if
                                 x < mean_cover_cont - std_cover_cont or x > mean_cover_cont + std_cover_cont]
            else:
                outliers_cont = []

            # Now, compute similar statistics for non-June data.
            non_june_subset = subset[subset['Month'] != 6]
            non_june_valid_binary = non_june_subset[non_june_subset['cover_frac_binary'] > 0].dropna(
                subset=['cover_frac_binary'])
            non_june_valid_cont = non_june_subset[non_june_subset['cover_frac_cont'] > 0].dropna(
                subset=['cover_frac_cont'])

            if len(non_june_valid_binary) > 1:
                nj_years_bin = np.array(non_june_valid_binary['Year']).reshape(-1, 1)
                nj_cover_bin = non_june_valid_binary['cover_frac_binary'].values
                nj_stats_bin = calculate_statistics2(nj_years_bin.flatten(), nj_cover_bin)
                non_june_slope_bin = nj_stats_bin.get('Slope', None)
                non_june_slope_se_bin = nj_stats_bin.get('Slope SE', None)
                non_june_r2_bin = nj_stats_bin.get('R²', None)
                non_june_ci_bin = nj_stats_bin.get('Confidence Interval', (None, None))
                non_june_rmse_bin = nj_stats_bin.get('RMSE', None)
            else:
                non_june_slope_bin = non_june_slope_se_bin = non_june_r2_bin = non_june_rmse_bin = np.nan
                non_june_ci_bin = (np.nan, np.nan)

            if len(non_june_valid_cont) > 1:
                nj_years_cont = np.array(non_june_valid_cont['Year']).reshape(-1, 1)
                nj_cover_cont = non_june_valid_cont['cover_frac_cont'].values
                nj_stats_cont = calculate_statistics2(nj_years_cont.flatten(), nj_cover_cont)
                non_june_slope_cont = nj_stats_cont.get('Slope', None)
                non_june_slope_se_cont = nj_stats_cont.get('Slope SE', None)
                non_june_r2_cont = nj_stats_cont.get('R²', None)
                non_june_ci_cont = nj_stats_cont.get('Confidence Interval', (None, None))
                non_june_rmse_cont = nj_stats_cont.get('RMSE', None)
            else:
                non_june_slope_cont = non_june_slope_se_cont = non_june_r2_cont = non_june_rmse_cont = np.nan
                non_june_ci_cont = (np.nan, np.nan)

            n_years = len(subset['Year'].unique())
            row0 = subset.iloc[0]

            results.append({
                'File': multisite_key,
                'Area': area,
                'Method': method,
                'n_years': n_years,
                # Binary-based statistics:
                'Mean Shrub Cover Binary': mean_cover_bin,
                'STD Shrub Cover Binary': std_cover_bin,
                'Slope Binary': slope_bin,
                'Slope SE Binary': slope_se_bin,
                'CI Lower Binary': ci_bin[0],
                'CI Upper Binary': ci_bin[1],
                'R2 Binary': r2_bin,
                'RMSE Binary': rmse_bin,
                'Outliers Binary': outliers_bin,
                'Slope Binary Ex June': non_june_slope_bin,
                'Slope SE Binary Ex June': non_june_slope_se_bin,
                'CI Lower Binary Ex June': non_june_ci_bin[0],
                'CI Upper Binary Ex June': non_june_ci_bin[1],
                'R2 Binary Ex June': non_june_r2_bin,
                'RMSE Binary Ex June': non_june_rmse_bin,
                # Continuous-based statistics:
                'Mean Shrub Cover Continuous': mean_cover_cont,
                'STD Shrub Cover Continuous': std_cover_cont,
                'Slope Continuous': slope_cont,
                'Slope SE Continuous': slope_se_cont,
                'CI Lower Continuous': ci_cont[0],
                'CI Upper Continuous': ci_cont[1],
                'R2 Continuous': r2_cont,
                'RMSE Continuous': rmse_cont,
                'Outliers Continuous': outliers_cont,
                'Slope Continuous Ex June': non_june_slope_cont,
                'Slope SE Continuous Ex June': non_june_slope_se_cont,
                'CI Lower Continuous Ex June': non_june_ci_cont[0],
                'CI Upper Continuous Ex June': non_june_ci_cont[1],
                'R2 Continuous Ex June': non_june_r2_cont,
                'RMSE Continuous Ex June': non_june_rmse_cont,
            })
    return results, df




import os
import numpy as np
import pandas as pd
from scipy import stats


# Your existing functions
def calculate_statistics2(x, y):
    # This function fits a linear regression model using statsmodels
    # and returns slope, slope SE, confidence interval, R², and RMSE.
    import statsmodels.api as sm
    x = np.array(x)
    y = np.array(y)
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    slope = results.params[1]
    intercept = results.params[0]
    r_squared = results.rsquared
    slope_se = results.bse[1]  # Standard error of the slope
    confidence_interval = results.conf_int(alpha=0.05)
    slope_ci_lower = confidence_interval[1, 0]
    slope_ci_upper = confidence_interval[1, 1]
    rmse = np.sqrt(results.mse_resid)
    return {
        'Slope': slope,
        'Intercept': intercept,
        'R²': r_squared,
        'Slope SE': slope_se,
        'Confidence Interval': (slope_ci_lower, slope_ci_upper),
        'RMSE': rmse
    }


def calculate_overall_significance_weighted(df, slope_column, se_column):
    """
    Calculate overall significance for a given slope column using a weighted mean (meta-analysis).

    Parameters:
      df (DataFrame): DataFrame containing the slopes and standard errors.
      slope_column (str): Column name for the slope values.
      se_column (str): Column name for the standard errors of the slopes.

    Returns:
      dict: Dictionary with the overall weighted mean slope, standard error, t-statistic, p-value, etc.
    """
    # Filter out rows with NaN, inf, or zero SE
    valid_df = df[np.isfinite(df[slope_column]) & np.isfinite(df[se_column])]
    valid_df = valid_df[valid_df[se_column] > 0]
    if len(valid_df) == 0:
        return {
            'Weighted Mean Slope': np.nan,
            'Standard Error': np.nan,
            't-Statistic': np.nan,
            'Degrees of Freedom': np.nan,
            'p-Value': np.nan,
            'Significant': False,
            'Number of Valid Slopes': 0
        }
    slopes = valid_df[slope_column].values
    ses = valid_df[se_column].values
    weights = 1 / (ses ** 2)
    weighted_mean_slope = np.sum(weights * slopes) / np.sum(weights)
    standard_error = np.sqrt(1 / np.sum(weights))
    t_stat = weighted_mean_slope / standard_error
    degrees_of_freedom = len(weights) - 1
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), degrees_of_freedom))
    significant = p_value < 0.05
    return {
        'Weighted Mean Slope': weighted_mean_slope,
        'Standard Error': standard_error,
        't-Statistic': t_stat,
        'Degrees of Freedom': degrees_of_freedom,
        'p-Value': p_value,
        'Significant': significant,
        'Number of Valid Slopes': len(valid_df)
    }


# Function to aggregate overall trend significance for each method
def aggregate_overall_trend_significance(results_df):
    """
    Given a DataFrame of per-area slopes and slope SE for various methods, compute the overall
    weighted trend for each method.

    The DataFrame is expected to have a column "Method" and columns for:
       - 'Slope Binary', 'Slope SE Binary'
       - 'Slope Continuous', 'Slope SE Continuous'
       - 'Slope Binary Ex June', 'Slope SE Binary Ex June'
       - 'Slope Continuous Ex June', 'Slope SE Continuous Ex June'

    Returns:
       A DataFrame with one row per method and columns for overall trends and significance.
    """
    methods = results_df['Method'].unique()
    agg_results = []
    for m in methods:
        df_m = results_df[results_df['Method'] == m]
        overall_bin = calculate_overall_significance_weighted(df_m, slope_column='Slope Binary',
                                                              se_column='Slope SE Binary')
        overall_cont = calculate_overall_significance_weighted(df_m, slope_column='Slope Continuous',
                                                               se_column='Slope SE Continuous')
        overall_bin_ex = calculate_overall_significance_weighted(df_m, slope_column='Slope Binary Ex June',
                                                                 se_column='Slope SE Binary Ex June')
        overall_cont_ex = calculate_overall_significance_weighted(df_m, slope_column='Slope Continuous Ex June',
                                                                  se_column='Slope SE Continuous Ex June')

        agg_results.append({
            'Method': m,
            'Weighted Mean Slope Binary': overall_bin['Weighted Mean Slope'],
            'Significant Binary': overall_bin['Significant'],
            'Weighted Mean Slope Continuous': overall_cont['Weighted Mean Slope'],
            'Significant Continuous': overall_cont['Significant'],
            'Weighted Mean Slope Binary Ex June': overall_bin_ex['Weighted Mean Slope'],
            'Significant Binary Ex June': overall_bin_ex['Significant'],
            'Weighted Mean Slope Continuous Ex June': overall_cont_ex['Weighted Mean Slope'],
            'Significant Continuous Ex June': overall_cont_ex['Significant']
        })
    return pd.DataFrame(agg_results)


# Function to pivot the aggregated results into a multi-index column structure
def pivot_trend_summary(agg_df):
    """
    Given an aggregated DataFrame with one row per method, pivot it so that the columns
    are arranged in a multi-level index with subcolumns for Binary and Continuous, both
    for the full dataset and for the non-June (Ex June) subset.

    For example, the final DataFrame might have columns:
       - ('Slope', 'Binary'), ('Slope', 'Continuous'),
         ('Slope Ex June', 'Binary'), ('Slope Ex June', 'Continuous'),
         ('Significance', 'Binary'), ('Significance', 'Continuous'),
         ('Significance Ex June', 'Binary'), ('Significance Ex June', 'Continuous')
    """
    # Create a new DataFrame with a multi-index for columns.
    pivot_data = []
    for idx, row in agg_df.iterrows():
        m = row['Method']
        pivot_data.append({
            'Method': m,
            ('Slope', 'Binary'): row['Weighted Mean Slope Binary'],
            ('Slope', 'Continuous'): row['Weighted Mean Slope Continuous'],
            ('Slope Ex June', 'Binary'): row['Weighted Mean Slope Binary Ex June'],
            ('Slope Ex June', 'Continuous'): row['Weighted Mean Slope Continuous Ex June'],
            ('Significance', 'Binary'): row['Significant Binary'],
            ('Significance', 'Continuous'): row['Significant Continuous'],
            ('Significance Ex June', 'Binary'): row['Significant Binary Ex June'],
            ('Significance Ex June', 'Continuous'): row['Significant Continuous Ex June']
        })
    # Convert to DataFrame
    pivot_df = pd.DataFrame(pivot_data)
    pivot_df = pivot_df.set_index('Method')
    # Set MultiIndex columns
    pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)
    return pivot_df


def flatten_columns(df):
    """
    Flatten a DataFrame with MultiIndex columns into single level by joining levels with an underscore.
    """
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    return df




# ALL SAVED SHEETS FOLDER
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225'

# block size 200px
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200'

# block size 200px 110225 added ms4 2018
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_110225_blocksize_200'

# blocksize 200, filtered edge artifacts
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/saved_sheets_blocksize_200_edge_artifact_230225'

# blocksize 200, filtered edge artifacts + CI from bootstrap 27-2-25
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225'



### WETTUNDRA
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra'


### LAKES
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers'


### LAKES sizes
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakessizes'


all_statistics = []
all_statistics2 = []
df_filtered_all2 = []

# # saving filtered excel files
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225_filtered2'
#
# # blocksize 200 filtered
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered'
#
# # saved_sheets_blocksize_200_extra_filtered_ms10a7
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_extra_filtered_ms10a9'

# saved_sheets_blocksize_200_extra_filtered_ms10a7
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9'
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9'
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225_extra_filtered_ms10a9'
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_CALIBRATED_blocksize_200_edge_artifact_CI_bootstrap_extra_filtered_ms10a9_230325'
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525'
output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT'
# output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525'
output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_060625_VIT'
output_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakessizes_FILTERED'
os.makedirs(output_dir, exist_ok=True)

valid_files = [file for file in os.listdir(folder_path) if not file.startswith('~$') and file.endswith('.xlsx')]

# for file2 in valid_files:
#     print(file2)


# ### TEST
# file_path = '/Users/radakovicd1/Downloads/test_shrub_cover_statistics_VIT_p70.xlsx'
# statistics_and_outliers = pd.read_excel(file_path, sheet_name='shrub_cover_statistics', engine='openpyxl')
#
# agg_df = aggregate_overall_trend_significance(statistics_and_outliers)
# # Pivot the aggregated results into a multi-index column DataFrame.
# trend_summary_df = pivot_trend_summary(agg_df)
# # (Optional) Flatten the multi-index columns if desired.
# trend_summary_df = flatten_columns(trend_summary_df)



# Loop over each Excel file in the folder.
for file in valid_files:
    if file.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file)
        # Load the sheet "shrub_cover_statistics" as a DataFrame.
        print(file)

        # Extract model name from the file name (e.g., remove the extension).
        model_name = file.split('_')[5]

        model_name_key = model_name  # or model_name.lower(), depending on how you set your dictionary keys
        calib = calibration_slopes.get(model_name_key, 1.0)
        print(model_name_key)
        print(calib)

        df = pd.read_excel(file_path, sheet_name='shrub_cover_data', engine='openpyxl')
        df['cover_frac_cont'] = df['cover_frac_cont'] / calib

        all_statistics = []
        df_filtered_all = []
        for multisite_key in site_years.keys():

            # df = pd.read_excel(file_path, sheet_name='shrub_cover_data')
            # df = pd.read_excel(file_path, sheet_name='shrub_cover_data', engine='openpyxl')
            # file_path = '/Users/radakovicd1/Downloads/test_shrub_cover_statistics_VIT_p70.xlsx'
            # statistics_and_outliers = pd.read_excel(file_path, sheet_name='shrub_cover_statistics', engine='openpyxl')

            # results, df_filtered = calculate_statistics_and_outliers_shrub_cover10feb2025(df, multisite_key=multisite_key, site_years=site_years)
            results, df_filtered = calculate_statistics_and_outliers_shrub_cover10feb2025(df, multisite_key=multisite_key, site_years=site_years, site_areas=site_areas)

            # You can then convert the results list to a DataFrame:
            statistics_and_outliers = pd.DataFrame(results)

            df_filtered_all.append(df_filtered)
            all_statistics.append(statistics_and_outliers)

        df_filtered_all2 = pd.concat(df_filtered_all, ignore_index=True)
        all_statistics_df = pd.concat(all_statistics, ignore_index=True)

        # Aggregate overall significance for each method.
        agg_df = aggregate_overall_trend_significance(all_statistics_df)
        # Pivot the aggregated results into a multi-index column DataFrame.
        trend_summary_df = pivot_trend_summary(agg_df)
        # (Optional) Flatten the multi-index columns if desired.
        trend_summary_df = flatten_columns(trend_summary_df)

        trend_summary_df['Model'] = model_name

        # (Optionally) Ensure "Method" is a column. If your pivoted DataFrame uses the index as Method, then:
        trend_summary_df = trend_summary_df.reset_index()  # This makes the former index a column called "Method"

        all_statistics2.append(trend_summary_df)

        savename_stats = file.replace('.xlsx', '_FILTERED2.xlsx')
        savename_stats2 = os.path.join(output_dir,savename_stats)
        with pd.ExcelWriter(savename_stats2) as writer:
            df_filtered_all2.to_excel(writer, sheet_name='shrub_cover_data', index=False)
            all_statistics_df.to_excel(writer, sheet_name='shrub_cover_statistics', index=False)
            trend_summary_df.to_excel(writer, sheet_name='trends', index=False)

# Combine all the individual DataFrames into one.
all_statistics_df2 = pd.concat(all_statistics2, ignore_index=True)




# all_statistics_df2.to_clipboard(index=False)  # Copies DataFrame without the index



### ---> use stats_extract_trends_excelfiles.py for better trends format




