"""
analysis_compas_actionrules.py – Comprehensive analysis for COMPAS ActionRules results
======================================================================================
Reads per-CF validation table and per-applicant summary from ActionRules pipeline.
Produces paper-level insights, plots, and statistical tests.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION – adjust file names if needed
# ----------------------------------------------------------------------
VALIDATION_FILE = "compas_validation_table_actionrules.csv"
SUMMARY_FILE = "compas_summary_actionrules.csv"
OUTPUT_PREFIX = "actionrules"   # for saving plots/reports

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df_cf = pd.read_csv(VALIDATION_FILE)
df_app = pd.read_csv(SUMMARY_FILE)

print(f"Loaded {len(df_cf)} counterfactuals from {df_cf['applicant_idx'].nunique()} applicants.")
print(f"Loaded summary for {len(df_app)} applicants.\n")

df_cf['is_valid'] = df_cf['conforms'] == True

# ----------------------------------------------------------------------
# 2. Rule violation breakdown
# ----------------------------------------------------------------------
def extract_rule(msg):
    if pd.isna(msg) or msg == "No violation":
        return None
    # Try to find a rule name ending with "Rule"
    match = re.search(r'(\w+Rule)', msg)
    if match:
        return match.group(0)
    # Fallback: take the first few words
    return msg.split(':')[0].strip()

df_cf['rule'] = df_cf['violations'].apply(lambda x: extract_rule(x) if x != "No violation" else None)

invalid_cfs = df_cf[~df_cf['is_valid']]
rule_counts = invalid_cfs['rule'].value_counts().dropna()
print("=== VIOLATION COUNTS PER RULE ===")
print(rule_counts)

# ----------------------------------------------------------------------
# 3. Compare valid vs invalid CFs on distance, cost, and feature values
# ----------------------------------------------------------------------
print("\n=== COMPARISON: VALID vs INVALID CFs ===")
valid = df_cf[df_cf['is_valid']]
invalid = df_cf[~df_cf['is_valid']]

# Metrics to compare
metrics = ['distance', 'cost']
features = ['age', 'priors_count', 'juv_other_count', 'days_b_screening_arrest', 'length_of_stay']

for metric in metrics + features:
    if metric not in df_cf.columns:
        continue
    v_mean = valid[metric].mean()
    inv_mean = invalid[metric].mean()
    stat, p = stats.mannwhitneyu(valid[metric], invalid[metric], alternative='two-sided')
    print(f"{metric}: Valid mean={v_mean:.2f}, Invalid mean={inv_mean:.2f}, p={p:.4f}")

# ----------------------------------------------------------------------
# 4. Feature change frequencies – cannot be computed without original values
# ----------------------------------------------------------------------
print("\nNOTE: To compute feature change frequencies, add original values to the validation table.")
print("Current table only contains counterfactual values.\n")

# ----------------------------------------------------------------------
# 5. Distribution of number of valid CFs per applicant
# ----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df_app['num_valid'], bins=range(0, 16, 1), kde=False)
plt.title("Number of Valid CFs per Applicant (ActionRules)")
plt.xlabel("Valid CFs")
plt.ylabel("Number of Applicants")
plt.savefig(f"valid_cfs_per_applicant_{OUTPUT_PREFIX}.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 6. Violation count per rule – bar plot
# ----------------------------------------------------------------------
if len(rule_counts) > 0:
    plt.figure(figsize=(10,6))
    rule_counts.sort_values(ascending=False).plot(kind='bar')
    plt.title("Violations per SHACL Rule (ActionRules)")
    plt.ylabel("Number of violations")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"violation_per_rule_{OUTPUT_PREFIX}.png", dpi=150)
    plt.show()

# ----------------------------------------------------------------------
# 7. Boxplot: distance by validity
# ----------------------------------------------------------------------
plt.figure(figsize=(6,5))
sns.boxplot(x='is_valid', y='distance', data=df_cf)
plt.title("Distance Distribution: Valid vs Invalid CFs (ActionRules)")
plt.xlabel("Valid")
plt.ylabel("Distance")
plt.savefig(f"distance_by_validity_{OUTPUT_PREFIX}.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 8. Statistical summary table (LaTeX-style)
# ----------------------------------------------------------------------
# Select only columns that exist
cols_to_agg = ['distance', 'cost'] + [f for f in features if f in df_cf.columns]
summary_stats = df_cf.groupby('is_valid')[cols_to_agg].agg(['mean', 'std', 'min', 'max']).round(2)

print("\n=== SUMMARY TABLE (Valid vs Invalid) ===")
print(summary_stats)

summary_stats.to_csv(f"summary_stats_{OUTPUT_PREFIX}.csv")
print(f"Summary stats saved to summary_stats_{OUTPUT_PREFIX}.csv")

# ----------------------------------------------------------------------
# 9. Correlation between distance and cost
# ----------------------------------------------------------------------
corr_dist_cost = df_cf[['distance', 'cost']].corr().iloc[0,1]
print(f"\nCorrelation between distance and cost: {corr_dist_cost:.3f}")

# ----------------------------------------------------------------------
# 10. Save a detailed text report
# ----------------------------------------------------------------------
with open(f"analysis_{OUTPUT_PREFIX}_report.txt", "w") as f:
    f.write("COUNTERFACTUAL ANALYSIS – COMPAS (ActionRules)\n")
    f.write("==================================================\n\n")
    f.write(f"Total CFs: {len(df_cf)}\n")
    f.write(f"Valid CFs: {valid.shape[0]} ({valid.shape[0]/len(df_cf):.2%})\n")
    f.write(f"Invalid CFs: {invalid.shape[0]} ({invalid.shape[0]/len(df_cf):.2%})\n\n")
    f.write("Violation counts:\n")
    f.write(rule_counts.to_string())
    f.write("\n\nComparison of valid vs invalid:\n")
    f.write(summary_stats.to_string())
    f.write(f"\n\nCorrelation distance-cost: {corr_dist_cost:.3f}\n")
    f.write("\nMann-Whitney tests:\n")
    for metric in metrics + features:
        if metric not in df_cf.columns:
            continue
        _, p = stats.mannwhitneyu(valid[metric], invalid[metric], alternative='two-sided')
        f.write(f"{metric}: p={p:.4f}\n")
print(f"Detailed report saved to analysis_{OUTPUT_PREFIX}_report.txt")