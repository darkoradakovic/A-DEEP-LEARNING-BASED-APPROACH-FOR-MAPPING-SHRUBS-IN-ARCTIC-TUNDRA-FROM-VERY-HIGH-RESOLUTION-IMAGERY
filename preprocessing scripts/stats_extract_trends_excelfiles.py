

# Darko Radakovic
# Montclair State University
# 23-03-25

### USE AFTER stats_filter_sites_results.py

### Extract trends from xlsx files from a dir

# Gives all the trends that can be copy pasted into excel (no excel spreadsheet saving output)


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
        # overall_bin_ex = calculate_overall_significance_weighted(df_m, slope_column='Slope Binary Ex June',
        #                                                          se_column='Slope SE Binary Ex June')
        # overall_cont_ex = calculate_overall_significance_weighted(df_m, slope_column='Slope Continuous Ex June',
        #                                                           se_column='Slope SE Continuous Ex June')

        agg_results.append({
            'Method': m,
            'Weighted Mean Slope Binary': overall_bin['Weighted Mean Slope'],
            'Significant Binary': overall_bin['Significant'],
            'Weighted Mean Slope Continuous': overall_cont['Weighted Mean Slope'],
            'Significant Continuous': overall_cont['Significant'],
            # 'Weighted Mean Slope Binary Ex June': overall_bin_ex['Weighted Mean Slope'],
            # 'Significant Binary Ex June': overall_bin_ex['Significant'],
            # 'Weighted Mean Slope Continuous Ex June': overall_cont_ex['Weighted Mean Slope'],
            # 'Significant Continuous Ex June': overall_cont_ex['Significant'],
            'p_value Slope Binary': overall_bin['p-Value'],
            'p_value Slope Cont': overall_cont['p-Value']
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
            # ('Slope Ex June', 'Binary'): row['Weighted Mean Slope Binary Ex June'],
            # ('Slope Ex June', 'Continuous'): row['Weighted Mean Slope Continuous Ex June'],
            ('Significance', 'Binary'): row['Significant Binary'],
            ('Significance', 'Continuous'): row['Significant Continuous'],
            # ('Significance Ex June', 'Binary'): row['Significant Binary Ex June'],
            ('p_value binary', 'Binary'): row['p_value Slope Binary'],
            ('p_value cont', 'Continuous'): row['p_value Slope Cont']
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


# folder_path = "/Users/radakovicd1/Downloads/saved_sheets"
folder_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_070225_filtered2"

# blocksize 200px and filtered
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_filtered'

# saved_sheets_blocksize_200_extra_filtered_ms10a7
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_extra_filtered_ms10a9'


# saved_sheets_blocksize_200_extra_filtered_ms10a7, extra ms4 2018
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_110225_extra_filtered_ms10a9'

# edge artifacts removed and block size (also extra_filtered_ms10a7, extra ms4 2018) 23-02-25
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_blocksize_200_edge_artifact_230225_extra_filtered_ms10a9'

# # blocksize 200, filtered edge artifacts + CI from bootstrap 27-2-25 (also extra_filtered_ms10a7)
# ## BAD NEGATIVE SLOPES ???
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets2_blocksize_200_edge_artifact_CI_bootstrap_270225'

## 23-3-25 Calibration
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_CALIBRATED_blocksize_200_edge_artifact_CI_bootstrap_extra_filtered_ms10a9_230325'



## Wettundra (basic)
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra'

## Wettundra extra filtered
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_040525'
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_wettundra_extra_filtered_ms10a9_060625_VIT'


## lakesrivers (basic)
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers'

## lakesrivers extra filtered
folder_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/saved_sheets_lakesrivers_extra_filtered_ms10a9_040525'



all_statistics = []

# Loop over each Excel file in the folder.
for file in sorted(os.listdir(folder_path)):
    if file.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file)
        # Load the sheet "shrub_cover_statistics" as a DataFrame.
        results_slope_overlap = pd.read_excel(file_path, sheet_name='shrub_cover_statistics')

        # Aggregate overall significance for each method.
        agg_df = aggregate_overall_trend_significance(results_slope_overlap)
        # Pivot the aggregated results into a multi-index column DataFrame.
        trend_summary_df = pivot_trend_summary(agg_df)
        # (Optional) Flatten the multi-index columns if desired.
        trend_summary_df = flatten_columns(trend_summary_df)

        # Extract model name from the file name (e.g., remove the extension).
        model_name = file.split('_')[5]
        # model_name = os.path.splitext(file)[0]
        trend_summary_df['Model'] = model_name

        # (Optionally) Ensure "Method" is a column. If your pivoted DataFrame uses the index as Method, then:
        trend_summary_df = trend_summary_df.reset_index()  # This makes the former index a column called "Method"

        all_statistics.append(trend_summary_df)

# Combine all the individual DataFrames into one.
all_statistics_df = pd.concat(all_statistics, ignore_index=True)

# # Finally, you can write this DataFrame to Excel:
# trend_summary_df.to_excel("overall_trend_summary.xlsx")

all_statistics_df.to_clipboard(index=False)  # Copies DataFrame without the index


