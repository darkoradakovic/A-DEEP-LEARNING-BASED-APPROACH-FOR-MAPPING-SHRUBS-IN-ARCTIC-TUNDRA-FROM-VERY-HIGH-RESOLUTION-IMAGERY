

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.ticker as ticker
import numpy as np


# shrub  model performance
file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_5_11_12.xlsx"

# wettundra model performance
file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_wettundra_validation_output_accuracy_f1_mae.xlsx'

# lakes model performance
file_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_lakes_validation_output_accuracy_f1_mae.xlsx'

savedir ='/Users/radakovicd1/Documents/Screenshots/'

# Load the Excel file
# df = pd.read_excel(file_path, sheet_name='with_f1_std')  # Loads all sheets into a dictionary
# df = pd.read_excel(file_path)
df = pd.read_excel(file_path, sheet_name='Sheet1')  # Loads all sheets into a dictionary



### TABLE OF ALL MODELS
# Compute mean per method of each model
# mean_df = df.groupby(["Model", "Method"]).mean(numeric_only=True).reset_index()
mean_df = df.groupby(["Model", "Method","Input data"]).mean(numeric_only=True).reset_index()

# mead_filtered_df = mean_df[(mean_df["Method"] == "MEAN_dyn_p70")]
mead_filtered_df = mean_df[(mean_df["Method"] == "MEAN_dyn_p80")]
# mead_filtered_df = mean_df[(mean_df["Method"] == "MEAN_dyn_p60")]
# mead_filtered_df = mean_df[(mean_df["Input data"] == "RAD + P1BS")]
# mead_filtered_df = mean_df.groupby(["Model"]).mean(numeric_only=True).reset_index()


# Format the values
formatted_data = mean_df.copy()
formatted_data[['Accuracy', 'Precision', 'Recall', 'F1']] = formatted_data[['Accuracy', 'Precision', 'Recall', 'F1']].mul(100).round(0).astype(int)  # Convert to percentage without decimals
formatted_data[['AUC', 'MAE', 'ME']] = formatted_data[['AUC', 'MAE', 'ME']].round(2)  # Round to 2 decimals

# Select relevant columns for display
table_data = formatted_data[['Model', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MAE', 'ME']]

# Display table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')

plt.show(block=True)

# mean_df.to_clipboard(index=False)
mead_filtered_df.to_clipboard(index=False)   # USE FOR CHAPTERS PHD




# ----------------------------------------------------------
### ------------ PLOT THRESHOLD PERCENTILES AGAINST ACCURACY (CHAPTER 2) ------------
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
## option 2 ONLY FOR SHRUBS
#shrub
file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_v2.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")


grouped = (
    df
    .groupby(["Model", "Method", "Input data"], as_index=False)
    .mean(numeric_only=True)
)

wanted_methods  = ["MEAN_dyn_p60", "MEAN_dyn_p70", "MEAN_dyn_p80"]
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "ViT"]

plot_df = grouped[
    (grouped["Method"].isin(wanted_methods)) &
    (grouped["Model"].isin(selected_models))
].copy()

# Ensure the “Method” column is categorical so we can re‐index in fixed order:
plot_df["Method"] = plot_df["Method"].astype("category")
plot_df["Method"].cat.set_categories(wanted_methods, inplace=True)
plot_df = plot_df.sort_values(["Method", "Model"])

# Convert to percentages:
plot_df["Accuracy_pct"] = plot_df["Accuracy"] * 100
plot_df["F1_pct"]       = plot_df["F1"] * 100

# Likewise scale the CI columns to percent:
plot_df["Accuracy_CI_lower_pct"] = plot_df["Accuracy_CI_lower"] * 100
plot_df["Accuracy_CI_upper_pct"] = plot_df["Accuracy_CI_upper"] * 100
plot_df["F1_CI_lower_pct"]       = plot_df["F1_CI_lower"] * 100
plot_df["F1_CI_upper_pct"]       = plot_df["F1_CI_upper"] * 100


# (a) Mean accuracy pivot:
pivot_acc_mean = (
    plot_df
    .pivot_table(
        index="Method",
        columns="Model",
        values="Accuracy_pct",
        aggfunc="mean"
    )
    .reindex(wanted_methods)
)

# (b) Accuracy_CI_lower pivot:
pivot_acc_low = (
    plot_df
    .pivot_table(
        index="Method",
        columns="Model",
        values="Accuracy_CI_lower_pct",
        aggfunc="mean"
    )
    .reindex(wanted_methods)
)

# (c) Accuracy_CI_upper pivot:
pivot_acc_high = (
    plot_df
    .pivot_table(
        index="Method",
        columns="Model",
        values="Accuracy_CI_upper_pct",
        aggfunc="mean"
    )
    .reindex(wanted_methods)
)

# (d) Mean F1 pivot:
pivot_f1_mean = (
    plot_df
    .pivot_table(
        index="Method",
        columns="Model",
        values="F1_pct",
        aggfunc="mean"
    )
    .reindex(wanted_methods)
)

# (e) F1_CI_lower & upper pivots:
pivot_f1_low = (
    plot_df
    .pivot_table(
        index="Method",
        columns="Model",
        values="F1_CI_lower_pct",
        aggfunc="mean"
    )
    .reindex(wanted_methods)
)
pivot_f1_high = (
    plot_df
    .pivot_table(
        index="Method",
        columns="Model",
        values="F1_CI_upper_pct",
        aggfunc="mean"
    )
    .reindex(wanted_methods)
)

# err_below = mean − CI_lower;  err_above = CI_upper − mean
err_acc_below = pivot_acc_mean - pivot_acc_low
err_acc_above = pivot_acc_high - pivot_acc_mean

err_f1_below = pivot_f1_mean - pivot_f1_low
err_f1_above = pivot_f1_high - pivot_f1_mean

# Plot 1 --- “Accuracy vs. Method” with error bars

x = np.arange(len(wanted_methods))  # [0,1,2]
wanted_methods = ['p60','p70','p80']
plt.figure(figsize=(8, 8))
for model in pivot_acc_mean.columns:
    y = pivot_acc_mean[model].values
    err_b = err_acc_below[model].values
    err_a = err_acc_above[model].values

    plt.errorbar(
        x, y,
        yerr=np.vstack([err_b, err_a]),
        marker='o',
        linestyle='-',
        linewidth=3,
        markersize=15,
        capsize=10,
        label=model
    )
plt.xticks(
    ticks=x,
    labels=wanted_methods,
    rotation=25,
    fontsize=20,
    fontweight='bold'
)
plt.xlabel("Threshold Method", fontsize=20, fontweight="bold")
plt.ylabel("Accuracy (%)",    fontsize=20, fontweight="bold")
# plt.title("Accuracy vs. Threshold by Model", fontsize=16, fontweight="bold")
# plt.ylim(60, 100)
plt.ylim(65, 95)
plt.xlim(-0.3, len(wanted_methods)-1 + 0.3)
# plt.xticks(rotation=25, fontsize=20, fontweight='bold')
# plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.grid(False)
plt.legend(loc="best", title="Model", fontsize=18, title_fontsize=18, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=1)
plt.tight_layout()
plt.show(block=True)


# Plot 2 --- “F1 Score vs. Method” with error bars
plt.figure(figsize=(8, 8))
for model in pivot_f1_mean.columns:
    y = pivot_f1_mean[model].values
    err_b = err_f1_below[model].values
    err_a = err_f1_above[model].values
    plt.errorbar(
        x, y,
        yerr=np.vstack([err_b, err_a]),
        marker='o',
        linestyle='-',
        linewidth=3,
        markersize=15,
        capsize=10,
        label=model
    )
plt.xticks(
    ticks=x,
    labels=wanted_methods,
    rotation=25,
    fontsize=20,
    fontweight='bold'
)
plt.xlabel("Threshold Method", fontsize=20, fontweight="bold")
plt.ylabel("F1 Score (%)",  fontsize=20, fontweight="bold")
# plt.title("F1 Score vs. Threshold by Model", fontsize=16, fontweight="bold")
# plt.ylim(0, 100)
plt.xlim(-0.3, len(wanted_methods)-1 + 0.3)
plt.yticks(fontsize=20, fontweight='bold')
plt.grid(False)
plt.legend(loc="best", title="Model", fontsize=18, title_fontsize=18, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=1)
plt.tight_layout()
plt.show(block=True)



#### stats percentiles vs accuracy/f1
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

print("=== Checking assumptions for ANOVA on Accuracy_pct by Method ===")

# 4.1.1 Normality within each Method
normality = {}
for m, grp in plot_df.groupby("Method"):
    stat, pval = stats.shapiro(grp["Accuracy_pct"])
    normality[m] = pval
    print(f"Shapiro‐Wilk for {m}: p = {pval:.3f}")

# 4.1.2 Levene’s test for equal variances
stat_levene, p_levene = stats.levene(
    *[grp["Accuracy_pct"].values for _, grp in plot_df.groupby("Method")]
)
print(f"Levene’s test for homogeneity of variance: p = {p_levene:.3f}")

# 4.1.3 Decide
if all(p > 0.05 for p in normality.values()) and (p_levene > 0.05):
    print("→ Using one‐way ANOVA (assumptions satisfied).")
else:
    print("→ Using Kruskal–Wallis test (assumptions violated).")


# Fit OLS: Accuracy_pct ~ C(Method)
model = smf.ols("Accuracy_pct ~ C(Method)", data=plot_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nOne‐way ANOVA table:")
print(anova_table)

## POST HOC
import scikit_posthocs as sp
print("\n=== Post‐hoc (Dunn) for Accuracy_pct across Method levels ===")
# Build the data for Dunn’s test:
data_all = plot_df["Accuracy_pct"].values
group_labels = plot_df["Method"].values
# Dunn’s test with Bonferroni correction
posthoc_acc = sp.posthoc_dunn(
    pd.DataFrame({"Accuracy_pct": data_all, "Method": group_labels}),
    val_col="Accuracy_pct",
    group_col="Method",
    p_adjust="bonferroni"
)
print(posthoc_acc.round(4))









##### THreshold vs accuracy
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0) Define file paths (replace with your real paths)
fp1 = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_5_8_10_11_12.xlsx"
fp2 = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_5_11_12.xlsx"
fp3 = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_11_12.xlsx"

df1 = pd.read_excel(fp1, sheet_name="Sheet1");  df1["Classes"] = "5_8_10_11_12"
df2 = pd.read_excel(fp2, sheet_name="Sheet1");  df2["Classes"] = "5_11_12"
df3 = pd.read_excel(fp3, sheet_name="Sheet1");  df3["Classes"] = "11_12"
df_all = pd.concat([df1, df2, df3], ignore_index=True).dropna()
# 1) Convert “Accuracy” and “F1” to percentages (0–100) and similarly scale their CI columns to %:
df_all["Accuracy_pct"]           = df_all["Accuracy"]           * 100.0
df_all["Accuracy_CI_lower_pct"]  = df_all["Accuracy_CI_lower"]  * 100.0
df_all["Accuracy_CI_upper_pct"]  = df_all["Accuracy_CI_upper"]  * 100.0
df_all["F1_pct"]                 = df_all["F1"]                 * 100.0
df_all["F1_CI_lower_pct"]        = df_all["F1_CI_lower"]        * 100.0
df_all["F1_CI_upper_pct"]        = df_all["F1_CI_upper"]        * 100.0

grouped = (
    df_all
    .groupby(["Classes","Method"], as_index=False)
    .mean(numeric_only=True)
    .loc[:, [
        "Classes","Method",
        "Accuracy_pct","Accuracy_CI_lower_pct","Accuracy_CI_upper_pct",
        "F1_pct","F1_CI_lower_pct","F1_CI_upper_pct"
    ]]
)

# 3) Pivot so that rows are indexed by (Classes, Method).  We will extract out
#    whatever unique Methods appear in these three sheets.  (We do not pre‐limit to p60/p70/p80.)
pivot_acc_mean = grouped.pivot(index=["Classes","Method"], columns=[], values="Accuracy_pct")
pivot_acc_low  = grouped.pivot(index=["Classes","Method"], columns=[], values="Accuracy_CI_lower_pct")
pivot_acc_high = grouped.pivot(index=["Classes","Method"], columns=[], values="Accuracy_CI_upper_pct")

pivot_f1_mean  = grouped.pivot(index=["Classes","Method"], columns=[], values="F1_pct")
pivot_f1_low   = grouped.pivot(index=["Classes","Method"], columns=[], values="F1_CI_lower_pct")
pivot_f1_high  = grouped.pivot(index=["Classes","Method"], columns=[], values="F1_CI_upper_pct")

# 4) Find all distinct “Classes” and all distinct “Method” values, in the order you want them to appear.
all_classes = ["5_8_10_11_12","5_11_12","11_12"]

# Get a sorted list of all Methods that actually appear under grouped["Method"]:
all_methods = sorted(grouped["Method"].unique().tolist())

# Build a list of (Classes,Method) tuples in the desired 2D order, so that we can reindex each pivot:
desired_tuples = []
for cls in all_classes:
    for meth in all_methods:
        desired_tuples.append((cls,meth))

pivot_acc_mean  = pivot_acc_mean. reindex(desired_tuples)
pivot_acc_low   = pivot_acc_low.   reindex(desired_tuples)
pivot_acc_high  = pivot_acc_high.  reindex(desired_tuples)
pivot_f1_mean   = pivot_f1_mean.   reindex(desired_tuples)
pivot_f1_low    = pivot_f1_low.    reindex(desired_tuples)
pivot_f1_high   = pivot_f1_high.   reindex(desired_tuples)
# 5) Compute error‐bar “below” and “above” lengths for each row:
err_acc_below = pivot_acc_mean.values - pivot_acc_low.values
err_acc_above = pivot_acc_high.values - pivot_acc_mean.values
err_f1_below  = pivot_f1_mean.values - pivot_f1_low.values
err_f1_above  = pivot_f1_high.values - pivot_f1_mean.values

# 6) Now we will produce two separate figures.  On each figure, the x‐axis ticks will be “all_methods”:
#    x positions: 0..(N_methods-1).  And for each “class” in all_classes, we will plot one error‐bar series
#    (with marker only, no connecting line) across those x positions.
N_methods = len(all_methods)
x = np.arange(N_methods)  # [0, 1, 2, ..., N_methods-1]
# For convenience, build a lookup from (Classes,Method) --> row index in our pivot arrays:
row_index_lookup = {desired_tuples[i]: i for i in range(len(desired_tuples))}

# all_methods = ['p60', 'p70', 'p80', 'Otsu', 'Yen']
# Plot #1: “Accuracy vs. Method” (one figure)
plt.figure(figsize=(8, 8))
for cls in all_classes:
    # collect three arrays of length N_methods:
    y_means = []
    y_err_lo = []
    y_err_hi = []
    for meth in all_methods:
        idx = row_index_lookup[(cls,meth)]
        y_means.append(pivot_acc_mean.values[idx])
        y_err_lo.append(err_acc_below[idx])
        y_err_hi.append(err_acc_above[idx])
    plt.errorbar(
        x,
        y_means,
        yerr=np.vstack([y_err_lo, y_err_hi]),
        fmt='s',
        # linestyle='-',
        # linewidth=3,
        markersize=15,
        capsize=10,
        label=cls
    )
plt.xticks(
    ticks=x,
    labels=all_methods,
    rotation=25,
    fontsize=20,
    fontweight='bold'
)
plt.xlabel("Threshold Method", fontsize=20, fontweight='bold')
plt.ylabel("Accuracy (%)",       fontsize=20, fontweight='bold')
# plt.title("Average Accuracy vs. Method (for each Class‐set)", fontsize=16, fontweight='bold')
# plt.ylim(0, 100)
plt.grid(False)
plt.yticks(fontsize=20, fontweight='bold')
# plt.legend(loc="best", title="Classes Used", fontsize=18, title_fontsize=18, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=1)
plt.legend(title="Classes Used", fontsize=18, title_fontsize=18, loc="upper left")
plt.tight_layout()
plt.show(block=True)


# Plot #2: “F1 Score vs. Method” (another figure)
plt.figure(figsize=(8, 8))
for cls in all_classes:
    y_means = []
    y_err_lo = []
    y_err_hi = []
    for meth in all_methods:
        idx = row_index_lookup[(cls,meth)]
        y_means.append(pivot_f1_mean.values[idx])
        y_err_lo.append(err_f1_below[idx])
        y_err_hi.append(err_f1_above[idx])
    plt.errorbar(
        x,
        y_means,
        yerr=np.vstack([y_err_lo, y_err_hi]),
        fmt='s',            # square marker only, no connecting line
        capsize=15,
        markersize=10,
        label=cls)
plt.xticks(
    ticks=x,
    labels=all_methods,
    rotation=25,
    fontsize=20,
    fontweight='bold')
plt.xlabel("Threshold Method", fontsize=20, fontweight='bold')
plt.ylabel("F1 Score (%)",       fontsize=20, fontweight='bold')
# plt.title("Average F1 Score vs. Method (for each Class‐set)", fontsize=16, fontweight='bold')
# plt.ylim(0, 100)
plt.yticks(fontsize=20, fontweight='bold')
plt.grid(False)
plt.legend(title="Classes Used", fontsize=18, title_fontsize=18, loc="best")
plt.tight_layout()
plt.show(block=True)



#### stats THRESHOLD METHODS vs Accuracy/F1
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp

# Ensure that “Classes” is treated as a categorical with a known order:
desired_order = ["5_8_10_11_12", "5_11_12", "11_12"]
df_all["Classes"] = df_all["Classes"].astype("category")
df_all["Classes"].cat.set_categories(desired_order, ordered=True, inplace=True)

for cls in desired_order:
    print(f"\n===== {cls} =====")
    sub = df_all[df_all["Classes"] == cls].copy()
    methods_in_class = sorted(sub["Method"].unique())

    # 1) SHAPIRO–WILK on Accuracy_pct for each Method
    print("\nShapiro–Wilk normality for Accuracy_pct by Method:")
    normality_acc = {}
    for m in methods_in_class:
        grp = sub[sub["Method"] == m]["Accuracy_pct"]
        if len(grp) >= 3:
            stat, pval = stats.shapiro(grp)
        else:
            # Too few samples to test; assign p=0 so we force non‐normal
            pval = 0.0
        normality_acc[m] = pval
        print(f"  Method {m:15s}: p = {pval:.3f}")
    # 2) LEVENE’s test across Method groups
    groups_acc = [sub[sub["Method"] == m]["Accuracy_pct"].values for m in methods_in_class]
    if all(len(g)>1 for g in groups_acc):
        levene_stat_acc, levene_p_acc = stats.levene(*groups_acc)
    else:
        levene_p_acc = 0.0
    print(f"Levene’s test for Accuracy_pct (among Methods): p = {levene_p_acc:.3f}")

    # 3) Decide ANOVA vs. Kruskal–Wallis
    if all(p > 0.05 for p in normality_acc.values()) and (levene_p_acc > 0.05):
        print("→ Using one‐way ANOVA for Accuracy_pct.")
        model_acc = smf.ols("Accuracy_pct ~ C(Method)", data=sub).fit()
        anova_acc = sm.stats.anova_lm(model_acc, typ=2)
        print("\nOne‐way ANOVA (Accuracy_pct) table:")
        print(anova_acc)
    else:
        print("→ Assumptions violated; using Kruskal–Wallis for Accuracy_pct.")
        H_acc, p_kw_acc = stats.kruskal(*groups_acc)
        print(f"Kruskal–Wallis H‐test for Accuracy_pct: H = {H_acc:.3f}, p = {p_kw_acc:.3f}")

    # 4) POST‐HOC DUNN for Accuracy_pct
    print("\nPost‐hoc Dunn’s test (Accuracy_pct) among Methods:")
    data_all_acc = sub["Accuracy_pct"].values
    group_labels_acc = sub["Method"].values
    posthoc_acc = sp.posthoc_dunn(
        pd.DataFrame({"Accuracy_pct": data_all_acc, "Method": group_labels_acc}),
        val_col="Accuracy_pct",
        group_col="Method",
        p_adjust="bonferroni"
    )
    print(posthoc_acc.round(4))


    # Stats F1
    # Repeat EXACTLY the same for F1_pct
    # 1) SHAPIRO–WILK on F1_pct for each Method
    print("\nShapiro–Wilk normality for F1_pct by Method:")
    normality_f1 = {}
    for m in methods_in_class:
        grp = sub[sub["Method"] == m]["F1_pct"]
        if len(grp) >= 3:
            stat, pval = stats.shapiro(grp)
        else:
            pval = 0.0
        normality_f1[m] = pval
        print(f"  Method {m:15s}: p = {pval:.3f}")
    # 2) LEVENE’s test for F1_pct
    groups_f1 = [sub[sub["Method"] == m]["F1_pct"].values for m in methods_in_class]
    if all(len(g)>1 for g in groups_f1):
        levene_stat_f1, levene_p_f1 = stats.levene(*groups_f1)
    else:
        levene_p_f1 = 0.0
    print(f"Levene’s test for F1_pct (among Methods): p = {levene_p_f1:.3f}")

    # 3) Decide ANOVA vs. Kruskal–Wallis
    if all(p > 0.05 for p in normality_f1.values()) and (levene_p_f1 > 0.05):
        print("→ Using one‐way ANOVA for F1_pct.")
        model_f1 = smf.ols("F1_pct ~ C(Method)", data=sub).fit()
        anova_f1 = sm.stats.anova_lm(model_f1, typ=2)
        print("\nOne‐way ANOVA (F1_pct) table:")
        print(anova_f1)
    else:
        print("→ Assumptions violated; using Kruskal–Wallis for F1_pct.")
        H_f1, p_kw_f1 = stats.kruskal(*groups_f1)
        print(f"Kruskal–Wallis H‐test for F1_pct: H = {H_f1:.3f}, p = {p_kw_f1:.3f}")

    # 4) POST‐HOC DUNN for F1_pct
    print("\nPost‐hoc Dunn’s test (F1_pct) among Methods:")
    data_all_f1 = sub["F1_pct"].values
    group_labels_f1 = sub["Method"].values
    posthoc_f1 = sp.posthoc_dunn(
        pd.DataFrame({"F1_pct": data_all_f1, "Method": group_labels_f1}),
        val_col="F1_pct",
        group_col="Method",
        p_adjust="bonferroni"
    )
    print(posthoc_f1.round(4))
    print("───────────────────────────────────────────────────────────\n")







#### PLOT CLASSES AGAINST ACCURACY and F1
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0) Define file paths (replace with your real paths)
fp1 = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_5_8_10_11_12.xlsx"
fp2 = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_5_11_12.xlsx"
fp3 = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_shrubs_validation_output_accuracy_f1_mae_class_11_12.xlsx"

# 1) Read each file, and add a column “Classes” that labels which class‐set was used
df1 = pd.read_excel(fp1, sheet_name="Sheet1")
df1["Classes"] = "5_8_10_11_12"
df2 = pd.read_excel(fp2, sheet_name="Sheet1")
df2["Classes"] = "5_11_12"
df3 = pd.read_excel(fp3, sheet_name="Sheet1")
df3["Classes"] = "11_12"
# 2) Concatenate all three into one DataFrame
df_all = pd.concat([df1, df2, df3], ignore_index=True)
df_all = df_all.dropna()
df_all["Accuracy_pct"]      = df_all["Accuracy"]      * 100
df_all["Accuracy_CI_lower_pct"] = df_all["Accuracy_CI_lower"] * 100
df_all["Accuracy_CI_upper_pct"] = df_all["Accuracy_CI_upper"] * 100
df_all["F1_pct"]            = df_all["F1"]            * 100
df_all["F1_CI_lower_pct"]       = df_all["F1_CI_lower"]       * 100
df_all["F1_CI_upper_pct"]       = df_all["F1_CI_upper"]       * 100
# 4) Group by (“Classes”, “Model”) and compute the mean of each numeric column
grouped = (
    df_all
    .groupby(["Classes", "Model"], as_index=False)
    .mean(numeric_only=True)
)
# 5) Pivot into wide tables: one for mean, one for CI_lower, one for CI_upper
pivot_acc_mean = (
    grouped
    .pivot(index="Classes", columns="Model", values="Accuracy_pct")
)
pivot_acc_low = (
    grouped
    .pivot(index="Classes", columns="Model", values="Accuracy_CI_lower_pct")
)
pivot_acc_high = (
    grouped
    .pivot(index="Classes", columns="Model", values="Accuracy_CI_upper_pct")
)
pivot_f1_mean = (
    grouped
    .pivot(index="Classes", columns="Model", values="F1_pct")
)
pivot_f1_low = (
    grouped
    .pivot(index="Classes", columns="Model", values="F1_CI_lower_pct")
)
pivot_f1_high = (
    grouped
    .pivot(index="Classes", columns="Model", values="F1_CI_upper_pct")
)
order = ["5_8_10_11_12", "5_11_12", "11_12"]
pivot_acc_mean = pivot_acc_mean.reindex(order)
pivot_acc_low  = pivot_acc_low.reindex(order)
pivot_acc_high = pivot_acc_high.reindex(order)
pivot_f1_mean = pivot_f1_mean.reindex(order)
pivot_f1_low  = pivot_f1_low.reindex(order)
pivot_f1_high = pivot_f1_high.reindex(order)
err_acc_below = pivot_acc_mean - pivot_acc_low
err_acc_above = pivot_acc_high - pivot_acc_mean
err_f1_below = pivot_f1_mean - pivot_f1_low
err_f1_above = pivot_f1_high - pivot_f1_mean
# 8) Prepare to plot:
models = pivot_acc_mean.columns.tolist()
x = np.arange(len(order))  # [0,1,2] for our three class‐sets

# Plot #1: “Accuracy vs. Classes Used for Validation” with 95 % CI errorbars
plt.figure(figsize=(8, 8))
for model in models:
    y      = pivot_acc_mean[model].values
    err_b  = err_acc_below[model].values
    err_a  = err_acc_above[model].values

    plt.errorbar(
        x, y,
        yerr=np.vstack([err_b, err_a]),
        marker="o", linestyle="-",
        linewidth=2, markersize=15,
        capsize=12,
        label=model
    )

plt.xticks(
    ticks=x,
    labels=order,
    rotation=20,
    fontsize=20, fontweight="bold"
)
plt.xlabel("Classes Used for Validation", fontsize=20, fontweight="bold")
plt.ylabel("Accuracy (%)",                fontsize=20, fontweight="bold")
# plt.title("Model Accuracy vs. Class Set",   fontsize=16, fontweight="bold")
plt.ylim(65, 90)   # or choose a narrower range if desired
plt.grid(False)
plt.legend(title="Model", fontsize=20, title_fontsize=20, loc="best")
plt.yticks(fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show(block=True)


# Plot #2: “F1 Score vs. Classes Used for Validation” with 95 % CI errorbars
plt.figure(figsize=(8, 8))
for model in models:
    y      = pivot_f1_mean[model].values
    err_b  = err_f1_below[model].values
    err_a  = err_f1_above[model].values
    plt.errorbar(
        x, y,
        yerr=np.vstack([err_b, err_a]),
        marker="s", linestyle="--",
        linewidth=2, markersize=8,
        capsize=5,
        label=model
    )
plt.xticks(
    ticks=x,
    labels=order,
    rotation=20,
    fontsize=20, fontweight="bold"
)
plt.xlabel("Classes Used for Validation", fontsize=20, fontweight="bold")
plt.ylabel("F1 Score (%)",              fontsize=20, fontweight="bold")
# plt.title("Model F1 Score vs. Class Set", fontsize=16, fontweight="bold")
# plt.ylim(0, 100)
plt.grid(False)
plt.legend(title="Model", fontsize=20, title_fontsize=20, loc="best")
plt.yticks(fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show(block=True)


#### stats CLASSES vs Accuracy/F1
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp  # for Dunn’s post‐hoc

# Make a working copy of df_all (already contains Accuracy_pct and F1_pct columns)
plot_df = df_all.copy()

# 1) Convert “Classes” to a categorical with the desired order—no inplace here:
desired_order = ["5_8_10_11_12", "5_11_12", "11_12"]
plot_df["Classes"] = plot_df["Classes"].astype("category")
plot_df["Classes"] = plot_df["Classes"].cat.set_categories(desired_order, ordered=True)

print("=== Checking assumptions for ANOVA on Accuracy_pct by Classes ===")

# 2) Shapiro–Wilk normality per class
normality_acc = {}
for cls, grp in plot_df.groupby("Classes"):
    stat, pval = stats.shapiro(grp["Accuracy_pct"])
    normality_acc[cls] = pval
    print(f"Shapiro–Wilk for {cls}: p = {pval:.3f}")

# 3) Levene’s test for equal variances
levene_stat_acc, levene_p_acc = stats.levene(
    *[grp["Accuracy_pct"].values for _, grp in plot_df.groupby("Classes")]
)
print(f"Levene’s test for homogeneity of variance (Accuracy_pct): p = {levene_p_acc:.3f}")

# 4) Choose test for Accuracy_pct
if all(p > 0.05 for p in normality_acc.values()) and (levene_p_acc > 0.05):
    print("→ Using one‐way ANOVA for Accuracy_pct (assumptions satisfied).")
    anova_acc = sm.stats.anova_lm(
        smf.ols("Accuracy_pct ~ C(Classes)", data=plot_df).fit(),
        typ=2
    )
    print("\nOne‐way ANOVA table for Accuracy_pct:")
    print(anova_acc)
else:
    print("→ Assumptions violated; using Kruskal–Wallis for Accuracy_pct.")
    groups_acc = [grp["Accuracy_pct"].values for _, grp in plot_df.groupby("Classes")]
    H_acc, p_kw_acc = stats.kruskal(*groups_acc)
    print(f"Kruskal–Wallis H‐test for Accuracy_pct: H = {H_acc:.3f}, p = {p_kw_acc:.3f}")

# 5) Post‐hoc Dunn’s test for Accuracy_pct (Bonferroni‐adjusted)
print("\n=== Post‐hoc (Dunn) for Accuracy_pct across Classes ===")
data_all_acc    = plot_df["Accuracy_pct"].values
group_labels_acc = plot_df["Classes"].values
posthoc_acc = sp.posthoc_dunn(
    pd.DataFrame({"Accuracy_pct": data_all_acc, "Classes": group_labels_acc}),
    val_col="Accuracy_pct",
    group_col="Classes",
    p_adjust="bonferroni"
)
print(posthoc_acc.round(4))


# --- Now repeat for F1_pct ---

# 6) Shapiro–Wilk normality per class (F1_pct)
normality_f1 = {}
for cls, grp in plot_df.groupby("Classes"):
    stat, pval = stats.shapiro(grp["F1_pct"])
    normality_f1[cls] = pval
    print(f"Shapiro–Wilk for {cls}: p = {pval:.3f}")

# 7) Levene’s test for F1_pct
levene_stat_f1, levene_p_f1 = stats.levene(
    *[grp["F1_pct"].values for _, grp in plot_df.groupby("Classes")]
)
print(f"Levene’s test for homogeneity of variance (F1_pct): p = {levene_p_f1:.3f}")

# 8) Choose test for F1_pct
if all(p > 0.05 for p in normality_f1.values()) and (levene_p_f1 > 0.05):
    print("→ Using one‐way ANOVA for F1_pct (assumptions satisfied).")
    anova_f1 = sm.stats.anova_lm(
        smf.ols("F1_pct ~ C(Classes)", data=plot_df).fit(),
        typ=2
    )
    print("\nOne‐way ANOVA table for F1_pct:")
    print(anova_f1)
else:
    print("→ Assumptions violated; using Kruskal–Wallis for F1_pct.")
    groups_f1 = [grp["F1_pct"].values for _, grp in plot_df.groupby("Classes")]
    H_f1, p_kw_f1 = stats.kruskal(*groups_f1)
    print(f"Kruskal–Wallis H‐test for F1_pct: H = {H_f1:.3f}, p = {p_kw_f1:.3f}")

# 9) Post‐hoc Dunn’s test for F1_pct
print("\n=== Post‐hoc (Dunn) for F1_pct across Classes ===")
data_all_f1      = plot_df["F1_pct"].values
group_labels_f1  = plot_df["Classes"].values
posthoc_f1 = sp.posthoc_dunn(
    pd.DataFrame({"F1_pct": data_all_f1, "Classes": group_labels_f1}),
    val_col="F1_pct",
    group_col="Classes",
    p_adjust="bonferroni"
)
print(posthoc_f1.round(4))


## STATS PER MODEL for CLASSES
# For reproducibility, let’s explicitly set the “Classes” ordering:
desired_classes = ["5_8_10_11_12", "5_11_12", "11_12"]
df_all["Classes"] = df_all["Classes"].astype("category")
df_all["Classes"] = df_all["Classes"].cat.set_categories(desired_classes, ordered=True)

# Make sure we don’t accidentally keep any NaNs in Accuracy_pct or F1_pct:
df_all = df_all.dropna(subset=["Classes", "Model", "Accuracy_pct", "F1_pct"])

# List of unique models in your DataFrame
all_models = df_all["Model"].unique().tolist()

# We will store results in a dictionary for later inspection, if desired
stats_results = {}

for model_name in all_models:
    print(f"\n\n===== MODEL: {model_name} =====")
    # Subset to just this one model
    df_model = df_all[df_all["Model"] == model_name].copy()

    # Make sure Classes is still categorical with our desired order
    df_model["Classes"] = df_model["Classes"].astype("category")
    df_model["Classes"] = df_model["Classes"].cat.set_categories(desired_classes, ordered=True)

    # Prepare a container to collect test results for this model
    stats_results[model_name] = {
        "Accuracy": {},
        "F1": {}
    }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP A: Perform Shapiro–Wilk normality test on Accuracy_pct for each Class
    print("** Shapiro–Wilk normality (Accuracy_pct) by class‐set **")
    normality_acc = {}
    for cls, grp in df_model.groupby("Classes"):
        stat, pval = stats.shapiro(grp["Accuracy_pct"])
        normality_acc[cls] = pval
        print(f"  Class {cls:12s} → Shapiro p = {pval:.3f}")
    stats_results[model_name]["Accuracy"]["normality_p"] = normality_acc

    # STEP B: Levene’s test for equality of variances (Accuracy_pct) across classes
    groups_acc = [grp["Accuracy_pct"].values for _, grp in df_model.groupby("Classes")]
    levene_stat_acc, levene_p_acc = stats.levene(*groups_acc)
    print(f"** Levene (Accuracy_pct) p = {levene_p_acc:.3f} **")
    stats_results[model_name]["Accuracy"]["levene_p"] = levene_p_acc

    # STEP C: Decide whether to use one‐way ANOVA or Kruskal–Wallis on Accuracy_pct
    if all(p > 0.05 for p in normality_acc.values()) and (levene_p_acc > 0.05):
        print("→ Using one‐way ANOVA on Accuracy_pct (assumptions satisfied).")
        formula_acc = "Accuracy_pct ~ C(Classes)"
        lm_acc = smf.ols(formula_acc, data=df_model).fit()
        anova_acc = sm.stats.anova_lm(lm_acc, typ=2)
        print("\nANOVA table for Accuracy_pct:")
        print(anova_acc)
        stats_results[model_name]["Accuracy"]["anova_table"] = anova_acc
    else:
        print("→ Using Kruskal–Wallis on Accuracy_pct (assumptions violated).")
        H_acc, p_kw_acc = stats.kruskal(*groups_acc)
        print(f"Kruskal–Wallis H = {H_acc:.3f}, p = {p_kw_acc:.3f}")
        stats_results[model_name]["Accuracy"]["kruskal_stat"] = H_acc
        stats_results[model_name]["Accuracy"]["kruskal_p"] = p_kw_acc

    # STEP D: Post‐hoc Dunn’s test (Accuracy_pct) with Bonferroni correction
    print("\n** Post‐hoc Dunn’s (Accuracy_pct) across classes **")
    data_all_acc = df_model["Accuracy_pct"].values
    group_labels_acc = df_model["Classes"].values
    posthoc_acc = sp.posthoc_dunn(
        pd.DataFrame({"Accuracy_pct": data_all_acc, "Classes": group_labels_acc}),
        val_col="Accuracy_pct",
        group_col="Classes",
        p_adjust="bonferroni"
    )
    print(posthoc_acc.round(4))
    stats_results[model_name]["Accuracy"]["posthoc_dunn"] = posthoc_acc

    # ─────────────────────────────────────────────────────────────────────────
    # Repeat the same for F1_pct:
    print("\n** Shapiro–Wilk normality (F1_pct) by class‐set **")
    normality_f1 = {}
    for cls, grp in df_model.groupby("Classes"):
        stat, pval = stats.shapiro(grp["F1_pct"])
        normality_f1[cls] = pval
        print(f"  Class {cls:12s} → Shapiro p = {pval:.3f}")
    stats_results[model_name]["F1"]["normality_p"] = normality_f1

    groups_f1 = [grp["F1_pct"].values for _, grp in df_model.groupby("Classes")]
    levene_stat_f1, levene_p_f1 = stats.levene(*groups_f1)
    print(f"** Levene (F1_pct) p = {levene_p_f1:.3f} **")
    stats_results[model_name]["F1"]["levene_p"] = levene_p_f1

    if all(p > 0.05 for p in normality_f1.values()) and (levene_p_f1 > 0.05):
        print("→ Using one‐way ANOVA on F1_pct (assumptions satisfied).")
        formula_f1 = "F1_pct ~ C(Classes)"
        lm_f1 = smf.ols(formula_f1, data=df_model).fit()
        anova_f1 = sm.stats.anova_lm(lm_f1, typ=2)
        print("\nANOVA table for F1_pct:")
        print(anova_f1)
        stats_results[model_name]["F1"]["anova_table"] = anova_f1
    else:
        print("→ Using Kruskal–Wallis on F1_pct (assumptions violated).")
        H_f1, p_kw_f1 = stats.kruskal(*groups_f1)
        print(f"Kruskal–Wallis H = {H_f1:.3f}, p = {p_kw_f1:.3f}")
        stats_results[model_name]["F1"]["kruskal_stat"] = H_f1
        stats_results[model_name]["F1"]["kruskal_p"] = p_kw_f1

    print("\n** Post‐hoc Dunn’s (F1_pct) across classes **")
    data_all_f1 = df_model["F1_pct"].values
    group_labels_f1 = df_model["Classes"].values
    posthoc_f1 = sp.posthoc_dunn(
        pd.DataFrame({"F1_pct": data_all_f1, "Classes": group_labels_f1}),
        val_col="F1_pct",
        group_col="Classes",
        p_adjust="bonferroni"
    )
    print(posthoc_f1.round(4))
    stats_results[model_name]["F1"]["posthoc_dunn"] = posthoc_f1










## Plot
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
# Convert Year to numeric format for sorting and plotting
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil"]

# df['Year'] = df['Year'].str.extract('(\d{4})').astype(int)
#
# # Plot accuracy over years
# plt.figure(figsize=(8, 5))
# sns.lineplot(data=df, x='Year', y='Accuracy', marker='o', hue='Model', linestyle='-', palette="tab10")
# plt.xlabel("Year")
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy Over Years")
# plt.xticks(rotation=45)
# plt.ylim(0, 1)  # Accuracy scale from 0 to 1
# plt.legend(title="Model")
# plt.grid(True)
# plt.show(block=True)




### SCATTERPLOT F1 score for the selected models using the segmentation method "MEAN_dyn_p70"
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
# df = pd.read_excel(file_path, sheet_name='with_f1_std_v2_CI')  # ONLY for shrubs

# Filter for the selected models and the specific segmentation method
selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]



# selected_models = ["VIT"]
filtered_df = df[(df["Model"].isin(selected_models)) & (df["Method"] == "MEAN_dyn_p80") & (df["Input data"] == "RAD + P1BS")]

# Create the plot
plt.figure(figsize=(10, 10))
# Scatter plot with error bars
for model in selected_models:
    model_data = filtered_df[filtered_df["Model"] == model]

    # Compute symmetric error: half the difference between upper and lower CI.
    ci_half = (model_data["F1_CI_upper"] - model_data["F1_CI_lower"]).abs() / 2.0

    # Plot with error bars and connecting lines.
    plt.errorbar(model_data["Year"], model_data["F1"]*100,
                 yerr=ci_half*100,
                 fmt='o-', label=model, capsize=10, alpha=0.8, markersize=15, linewidth=3, elinewidth=2, capthick=2)

plt.xlabel("Year", fontsize=20, fontweight='bold')
plt.ylabel("F1 Score (%)", fontsize=20, fontweight='bold')
# plt.title("F1 Score over years using \n p80 threshold for tall shrub predictions \n with 95% Confidence Intervals",
#           fontsize=26, fontweight='bold')
plt.xticks(rotation=20, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
# plt.ylim(0.3 , 0.7)
# plt.ylim(30 , 70) # shrubs
# plt.ylim(30 , 70) # shrubs
plt.legend(fontsize=20,loc="best", labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=1)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.show(block=True)
plt.tight_layout()
plt.show(block=True)

# filename = 'models_F1_over_years_v3.jpg'
# filename2 = os.path.join(savedir,filename)
# plt.savefig(filename2)
# plt.close()


### SCATTERPLOT Accuracy for the selected models using the segmentation method "MEAN_dyn_p70"
# Create the plot
# plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(figsize=(10, 10))
# Scatter plot with error bars
for model in selected_models:
    model_data = filtered_df[filtered_df["Model"] == model]

    # Compute symmetric error: half the difference between upper and lower CI.
    ci_half = (model_data["Accuracy_CI_upper"] - model_data["Accuracy_CI_lower"]).abs() / 2.0

    # Plot with error bars and connecting lines.
    plt.errorbar(model_data["Year"], model_data["Accuracy"]*100,
                 yerr=ci_half*100,
                 fmt='o-', label=model, capsize=10, alpha=0.8, markersize=15, linewidth=3, elinewidth=2, capthick=2)

plt.xlabel("Year", fontsize=20, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=20, fontweight='bold')
# plt.title("Accuracy over years using \n p80 threshold for tall shrub predictions \n with 95% Confidence Intervals",
#           fontsize=26, fontweight='bold')
plt.xticks(rotation=20, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
# plt.ylim(0.75 , 0.95)
# plt.ylim(75 , 95)  # shrubs
plt.legend(fontsize=20,loc="best", labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=1)
# plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show(block=True)






# ### BARPLOT F1 score for the selected models using the segmentation method "MEAN_dyn_p70"
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# file_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_shrub_detect_metrics/model_validation_output_accuracy_f1_mae.xlsx"
#
# # df = pd.read_excel(file_path, sheet_name='with_f1_std')
# df = pd.read_excel(file_path, sheet_name='with_f1_std_v2_CI')
#
# selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
#
# # Assume filtered_df is your filtered DataFrame with columns: Year, Model, F1, F1_boot_std, etc.
# # Make sure Year is treated as a categorical variable (string)
# filtered_df = df[(df["Model"].isin(selected_models)) & (df["Method"] == "MEAN_dyn_p80") & (df["Input data"] == "RAD + P1BS")]
#
# # filtered_df.loc[:, "Year"] = filtered_df["Year"].astype(str)
#
#
#
# # Get unique years sorted in order.
# years = sorted(filtered_df["Year"].unique())
# n_years = len(years)
# n_models = len(selected_models)
#
# # Define bar width and positions for each group.
# bar_width = 0.15
# x = np.arange(n_years)  # positions for the groups
#
# fig, ax = plt.subplots(figsize=(12, 10))
#
# for idx, model in enumerate(selected_models):
#     f1_vals = []
#     f1_err = []  # half CI (error bar)
#     for year in years:
#         # Filter for this model and year.
#         model_data = filtered_df[(filtered_df["Year"] == year) & (filtered_df["Model"] == model)]
#         if not model_data.empty:
#             mean_f1 = model_data["F1"].mean()
#             # Calculate half the width of the 95% CI.
#             ci_lower = model_data["F1_CI_lower"].mean()
#             ci_upper = model_data["F1_CI_upper"].mean()
#             mean_err = (ci_upper - ci_lower) / 2
#             # Check that error is finite.
#             if not np.isfinite(mean_err):
#                 mean_err = 0
#             f1_vals.append(mean_f1)
#             f1_err.append(mean_err)
#         else:
#             f1_vals.append(np.nan)
#             f1_err.append(0)
#
#     # Calculate x positions for bars for this model.
#     offset = (idx - (n_models - 1) / 2) * bar_width
#     bar_positions = x + offset
#     ax.bar(bar_positions, f1_vals, width=bar_width, yerr=f1_err, capsize=5, label=model, alpha=0.8)
#
# ax.set_xlabel("Year", fontsize=20, fontweight='bold')
# ax.set_ylabel("F1 Score", fontsize=20, fontweight='bold')
# ax.set_title("Models F1 Score over the Years \n using MEAN_dyn_p80", fontsize=26, fontweight='bold')
# ax.set_xticks(x)
# plt.xticks(fontsize=20, fontweight='bold')
# plt.yticks(fontsize=20, fontweight='bold')
# ax.set_xticklabels(years, rotation=35)
# ax.set_ylim(0, 1)
# ax.legend(title="Model", fontsize=20)
# ax.grid(True, linestyle="--", alpha=0.6)
#
# plt.show(block=True)










### SCATTERPLOT shrub cover vs ACCURACY for the selected models using the segmentation method "MEAN_dyn_p70"
import numpy as np
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
df = pd.read_excel(file_path, sheet_name='with_f1_std_v2_CI')  # Loads all sheets into a dictionary

selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
# selected_models = ["VIT"]


filtered_df = df[(df["Model"].isin(selected_models)) & (df["Method"] == "MEAN_dyn_p80") & (df["Input data"] == "RAD + P1BS")]

model_markers = {
    "CNN 512fil": "o",
    "Resnet50": "s",
    "VGG19": "X",
    "UNET 256fil": "D",
    "VIT": "^"
}

year_colors = {
    "QB02_2009": "orange",
    "WV02_2011": "blue",
    "QB02_2013": "green",
    "WV03_2016": "magenta",
    "WV03_2017": "red"
}


## with errorbars
import numpy as np
import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(10, 10))

# We'll store handles for separate legends
model_handles = {}
year_handles = {}

for idx, row in filtered_df.iterrows():
    model = row["Model"]
    year = row["Year"]
    ### -----> CHANGE -- ###
    shrub_cover = row["Cover_binary"]
    # shrub_cover = row["Cover_cont"]
    accuracy = row["Accuracy"]
    # Compute the error bar as half the CI range
    acc_lower = row["Accuracy_CI_lower"]
    acc_upper = row["Accuracy_CI_upper"]
    if np.isfinite(acc_lower) and np.isfinite(acc_upper):
        acc_err = (acc_upper - acc_lower) / 2.0
    else:
        acc_err = 0
    # Get marker shape & color
    marker = model_markers.get(model, "o")
    color = year_colors.get(year, "black")
    # Plot with error bar in y direction
    sc = ax.errorbar(
        shrub_cover*100, accuracy*100, yerr=acc_err*100,
        marker=marker, mfc=color, mec=color,
        ecolor=color, alpha=0.8, capsize=10, linestyle="none", markersize=15
    )
    # Create dummy handles for legends if needed
    if model not in model_handles:
        model_handles[model] = ax.plot([], [], marker=marker, color="black", label=model, linestyle="none")[0]
    if year not in year_handles:
        year_handles[year] = ax.plot([], [], marker="o", color=color, label=year, linestyle="none")[0]
# Two separate legends
model_legend = ax.legend(handles=model_handles.values(), loc="lower left", fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
ax.add_artist(model_legend)
year_legend = ax.legend(handles=year_handles.values(), loc="upper right", fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
ax.add_artist(year_legend)
ax.set_xlabel("Shrub Cover (%)", fontsize=20, fontweight='bold')
# ax.set_xlabel("Wet Tundra Cover (%)", fontsize=20, fontweight='bold')
# ax.set_xlabel("Surface water bodies Cover (%)", fontsize=20, fontweight='bold')
ax.set_ylabel("Accuracy (%)", fontsize=20, fontweight='bold')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
# ax.set_title("Model Accuracy vs. Tall Shrub Cover  \n using p80 as threshold of predictions  \n with 95% Confidence Intervals", fontsize=26, fontweight='bold')
# ax.set_title("Model Accuracy vs. Tall Shrub Continuous Cover \n with 95% Confidence Intervals", fontsize=18, fontweight='bold')
# ax.grid(True, linestyle="--", alpha=0.6)
# plt.ylim(0.7 , 1)
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.ylim(0.75 , 0.95)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
# plt.ylim(75 , 95)  # shrubs
plt.tight_layout()
plt.show(block=True)





### SCATTERPLOT shrub cover vs F1 for the selected models using the segmentation method "MEAN_dyn_p70"
import numpy as np
# ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
 # ───────────────────────────────────────────────────────────────────────── # ─────────────────────────────────────────────────────────────────────────
df = pd.read_excel(file_path, sheet_name='with_f1_std_v2_CI')  # Loads all sheets into a dictionary

selected_models = ["CNN 512fil", "Resnet50", "VGG19", "UNET 256fil", "VIT"]
# selected_models = ["VIT"]


filtered_df = df[(df["Model"].isin(selected_models)) & (df["Method"] == "MEAN_dyn_p80") & (df["Input data"] == "RAD + P1BS")]


model_markers = {
    "CNN 512fil": "o",
    "Resnet50": "s",
    "VGG19": "X",
    "UNET 256fil": "D",
    "VIT": "^"
}
year_colors = {
    "QB02_2009": "orange",
    "WV02_2011": "blue",
    "QB02_2013": "green",
    "WV03_2016": "magenta",
    "WV03_2017": "red"
}
## with errorbars
import numpy as np

fig, ax = plt.subplots(figsize=(10, 10))
# We'll store handles for separate legends
model_handles = {}
year_handles = {}

for idx, row in filtered_df.iterrows():
    model = row["Model"]
    year = row["Year"]
    ### -----> CHANGE -- ###
    shrub_cover = row["Cover_binary"]
    # shrub_cover = row["Cover_cont"]
    accuracy = row["F1"]
    # Compute the error bar as half the CI range
    acc_lower = row["F1_CI_lower"]
    acc_upper = row["F1_CI_upper"]
    if np.isfinite(acc_lower) and np.isfinite(acc_upper):
        acc_err = (acc_upper - acc_lower) / 2.0
    else:
        acc_err = 0
    # Get marker shape & color
    marker = model_markers.get(model, "o")
    color = year_colors.get(year, "black")
    # Plot with error bar in y direction
    sc = ax.errorbar(
        shrub_cover*100, accuracy*100, yerr=acc_err*100,
        marker=marker, mfc=color, mec=color,
        ecolor=color, alpha=0.8, capsize=10, linestyle="none", markersize=15
    )
    # Create dummy handles for legends if needed
    if model not in model_handles:
        model_handles[model] = ax.plot([], [], marker=marker, color="black", label=model, linestyle="none")[0]
    if year not in year_handles:
        year_handles[year] = ax.plot([], [], marker="o", color=color, label=year, linestyle="none")[0]

# Two separate legends
model_legend = ax.legend(handles=model_handles.values(), loc="best", fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
ax.add_artist(model_legend)
year_legend = ax.legend(handles=year_handles.values(), loc="lower center", fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
ax.add_artist(year_legend)
ax.set_xlabel("Shrub Cover (%)", fontsize=20, fontweight='bold')
# ax.set_xlabel("Wet Tundra Cover (%)", fontsize=20, fontweight='bold')
# ax.set_xlabel("Surface water bodies Cover (%)", fontsize=20, fontweight='bold')
ax.set_ylabel("F1 Score (%)", fontsize=20, fontweight='bold')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
# ax.set_title("Model F1 Score vs. Tall Shrub Cover \n using p80 as threshold of predictions \n with 95% Confidence Intervals", fontsize=26, fontweight='bold')
# ax.set_title("Model F1 Score vs. Tall Shrub Continuous Cover \n with 95% Confidence Intervals", fontsize=18, fontweight='bold')
# ax.grid(True, linestyle="--", alpha=0.6)
# plt.ylim(0.3 , 0.7)
# plt.ylim(30 , 70)  # shrubs
plt.tight_layout()
plt.show(block=True)






### STATSISTICS
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Assume you have a DataFrame with your model performance results,
# with columns for cover fraction (e.g., 'cover_frac_cont'), Accuracy, and F1 score.
# For this example, we'll use 'cover_frac_cont' for shrub cover and 'Accuracy' and 'F1' for performance.

# Remove rows with missing values in the columns of interest.
df_valid = filtered_df.dropna(subset=['Cover_binary', 'Accuracy', 'F1'])

# Calculate Pearson correlation for Accuracy vs. Shrub Cover.
r_acc, p_acc = pearsonr(df_valid['Cover_binary'], df_valid['Accuracy'])
# Calculate Pearson correlation for F1 vs. Shrub Cover.
r_f1, p_f1 = pearsonr(df_valid['Cover_binary'], df_valid['F1'])

print(f"Pearson correlation (Accuracy vs. Cover): r = {r_acc:.2f}, p = {p_acc:.3f}")
print(f"Pearson correlation (F1 vs. Cover): r = {r_f1:.2f}, p = {p_f1:.3f}")

# Pearson correlation (Accuracy vs. Cover): r = -0.85, p = 0.000
# Pearson correlation (F1 vs. Cover): r = -0.54, p = 0.005

# # Alternatively, if the data are not normally distributed, use Spearman correlation:
# r_acc_sp, p_acc_sp = spearmanr(df_valid['Cover_binary'], df_valid['Accuracy'])
# r_f1_sp, p_f1_sp = spearmanr(df_valid['Cover_binary'], df_valid['F1'])
#
# print(f"Spearman correlation (Accuracy vs. Cover): r = {r_acc_sp:.2f}, p = {p_acc_sp:.3f}")
# print(f"Spearman correlation (F1 vs. Cover): r = {r_f1_sp:.2f}, p = {p_f1_sp:.3f}")
#








### SCATTERPLOT shrub cover vs YEARS


