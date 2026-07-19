"""
analysis_paper.py – Comprehensive analysis for counterfactual evaluation
=======================================================================
Reads both per-applicant summary and per-CF validation table.
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
# 1. Load data
# ----------------------------------------------------------------------
df_cf = pd.read_csv("all_cf_validation_table.csv")
df_app = pd.read_csv("evaluation_summary.csv")

print(f"Loaded {len(df_cf)} counterfactuals from {df_cf['applicant_idx'].nunique()} applicants.")
print(f"Loaded summary for {len(df_app)} applicants.\n")

# Add a validity flag (True/False)
df_cf['is_valid'] = df_cf['conforms'] == True

# ----------------------------------------------------------------------
# 2. Rule violation breakdown
# ----------------------------------------------------------------------
# Extract rule names from violation messages (assuming format "RuleName: message")
def extract_rule(msg):
    if pd.isna(msg) or msg == "No violation":
        return None
    # Try to find the rule name: often starts with "ex:" or is a camel-case name
    # We'll look for any word that ends with "Rule" or starts with "ex:"
    match = re.search(r'(ex:\w+Rule|\w+Rule)', msg)
    if match:
        return match.group(0)
    # If not, take the first few words
    return msg.split(':')[0].strip()

df_cf['rule'] = df_cf['violations'].apply(lambda x: extract_rule(x) if x != "No violation" else None)

# Count violations per rule (only for invalid CFs)
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

for metric in ['distance', 'cost']:
    v_mean = valid[metric].mean()
    inv_mean = invalid[metric].mean()
    # Mann-Whitney U test (non-parametric)
    stat, p = stats.mannwhitneyu(valid[metric], invalid[metric], alternative='two-sided')
    print(f"{metric}: Valid mean={v_mean:.2f}, Invalid mean={inv_mean:.2f}, p={p:.4f}")

# Also compare age, credit_amount, duration (example features)
features_to_compare = ['age', 'credit_amount', 'duration']
for feat in features_to_compare:
    if feat in df_cf.columns:
        v_mean = valid[feat].mean()
        inv_mean = invalid[feat].mean()
        stat, p = stats.mannwhitneyu(valid[feat], invalid[feat], alternative='two-sided')
        print(f"{feat}: Valid mean={v_mean:.2f}, Invalid mean={inv_mean:.2f}, p={p:.4f}")

# ----------------------------------------------------------------------
# 4. Feature change frequencies (which features are most often changed?)
# ----------------------------------------------------------------------

print("\nWARNING: Feature change frequencies cannot be computed from this table.")
print("To get that, modify the pipeline to include original values in the per-CF CSV.")

# ----------------------------------------------------------------------
# 5. Distribution of number of valid CFs per applicant
# ----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df_app['num_valid'], bins=range(0, 16, 1), kde=False)
plt.title("Number of Valid CFs per Applicant")
plt.xlabel("Valid CFs")
plt.ylabel("Number of Applicants")
plt.savefig("valid_cfs_per_applicant.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 6. Violation count per rule – bar plot
# ----------------------------------------------------------------------
if len(rule_counts) > 0:
    plt.figure(figsize=(10,6))
    rule_counts.sort_values(ascending=False).plot(kind='bar')
    plt.title("Violations per SHACL Rule")
    plt.ylabel("Number of violations")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("violation_per_rule.png", dpi=150)
    plt.show()

# ----------------------------------------------------------------------
# 7. Boxplot: distance by validity
# ----------------------------------------------------------------------
plt.figure(figsize=(6,5))
sns.boxplot(x='is_valid', y='distance', data=df_cf)
plt.title("Distance Distribution: Valid vs Invalid CFs")
plt.xlabel("Valid")
plt.ylabel("Distance")
plt.savefig("distance_by_validity.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 8. Statistical summary table (LaTeX-style)
# ----------------------------------------------------------------------
summary_stats = df_cf.groupby('is_valid').agg({
    'distance': ['mean', 'std', 'min', 'max'],
    'cost': ['mean', 'std', 'min', 'max'],
    'credit_amount': ['mean', 'std'],
    'duration': ['mean', 'std'],
    'age': ['mean', 'std']
}).round(2)

print("\n=== SUMMARY TABLE (Valid vs Invalid) ===")
print(summary_stats)

# Save to CSV for LaTeX import
summary_stats.to_csv("summary_stats.csv")
print("Summary stats saved to summary_stats.csv")

# ----------------------------------------------------------------------
# 9. Additional: correlation between distance and cost (overall)
# ----------------------------------------------------------------------
corr_dist_cost = df_cf[['distance', 'cost']].corr().iloc[0,1]
print(f"\nCorrelation between distance and cost: {corr_dist_cost:.3f}")

# ----------------------------------------------------------------------
# 10. Save a detailed text report
# ----------------------------------------------------------------------
with open("analysis_paper_report.txt", "w") as f:
    f.write("COUNTERFACTUAL ANALYSIS – PAPER LEVEL\n")
    f.write("=====================================\n\n")
    f.write(f"Total CFs: {len(df_cf)}\n")
    f.write(f"Valid CFs: {valid.shape[0]} ({valid.shape[0]/len(df_cf):.2%})\n")
    f.write(f"Invalid CFs: {invalid.shape[0]} ({invalid.shape[0]/len(df_cf):.2%})\n\n")
    f.write("Violation counts:\n")
    f.write(rule_counts.to_string())
    f.write("\n\nComparison of valid vs invalid:\n")
    f.write(summary_stats.to_string())
    f.write(f"\n\nCorrelation distance-cost: {corr_dist_cost:.3f}\n")
    f.write("\nMann-Whitney tests:\n")
    for metric in ['distance', 'cost']:
        _, p = stats.mannwhitneyu(valid[metric], invalid[metric], alternative='two-sided')
        f.write(f"{metric}: p={p:.4f}\n")
print("detailed report stored to this")
