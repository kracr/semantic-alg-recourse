"""
analysis_paper.py – Comprehensive analysis for counterfactual evaluation
=======================================================================
Reads both per-applicant summary and per-CF validation table.
Produces paper-level insights, plots, and statistical tests.

UPDATED: Splits multi-rule violation messages and maps to SHACL rule names.
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
# 2. Rule violation breakdown – SPLIT AND MAP
# ----------------------------------------------------------------------
# Mapping from exact violation message fragment to SHACL rule name
FRAGMENT_TO_RULE = {
    "Savings cannot decrease by more than 1 level.": "SavingsDirectionalRule",
    "Employment tenure cannot decrease.": "EmploymentDirectionalRule",
    "Duration reduction exceeds 36 months.": "DurationReductionRule",
    "Loan amount reduced by > 70%.": "LoanReductionRule",
    "Checking status jump from <0 to >=200 DM is too large.": "CheckingJumpRule",
    "Employment tenure increase requires age increase.": "EmploymentAgeCohesionRule",
    "Savings increase requires age increase.": "SavingsAgeCohesionRule",
    "Savings jump from <100 to >=1000 DM is unrealistic.": "SavingsJumpRule",
}

def split_violations(violation_str):
    """Split a multi-rule violation string into a list of individual rule messages."""
    if pd.isna(violation_str) or violation_str == "No violation":
        return []
    # Split by ';' and strip whitespace
    parts = [p.strip() for p in violation_str.split(';') if p.strip()]
    return parts

# Apply to all rows (including valid ones – they have "No violation")
df_cf['violation_parts'] = df_cf['violations'].apply(split_violations)

# Explode the list into separate rows
df_rules = df_cf[~df_cf['is_valid']].explode('violation_parts')

# Map each fragment to rule name
df_rules['rule_name'] = df_rules['violation_parts'].map(FRAGMENT_TO_RULE)

# Count violations per rule
rule_counts = df_rules['rule_name'].value_counts().dropna()
print("=== VIOLATION COUNTS PER SHACL RULE (CORRECTED) ===")
print(rule_counts)

# For backwards compatibility, also keep a column with the first rule (optional)
# But we no longer need the old extract_rule.

# ----------------------------------------------------------------------
# 3. Compare valid vs invalid CFs on distance, cost, and feature values
# ----------------------------------------------------------------------
print("\n=== COMPARISON: VALID vs INVALID CFs ===")
valid = df_cf[df_cf['is_valid']]
invalid = df_cf[~df_cf['is_valid']]

for metric in ['distance', 'cost']:
    v_mean = valid[metric].mean()
    inv_mean = invalid[metric].mean()
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
# 4. Feature change frequencies (if original values exist in the table)
# ----------------------------------------------------------------------
# Check if original columns exist (e.g., 'orig_credit_amount')
orig_cols = [c for c in df_cf.columns if c.startswith('orig_')]
if orig_cols:
    print("\n=== FEATURE CHANGE FREQUENCIES ===")
    # For each feature with an original counterpart, compute how often it changed
    feature_changes = {}
    for orig_col in orig_cols:
        feat = orig_col.replace('orig_', '')
        if feat in df_cf.columns:
            changed = (df_cf[feat] != df_cf[orig_col]).sum()
            feature_changes[feat] = changed
    # Print sorted
    for feat, count in sorted(feature_changes.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat}: changed in {count} CFs ({count/len(df_cf):.2%})")
else:
    print("\nWARNING: Feature change frequencies cannot be computed because original values are not present.")
    print("To enable this, add original_* columns to the per-CF CSV in the pipeline.")

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
# 6. Violation count per rule – bar plot (using corrected counts)
# ----------------------------------------------------------------------
if len(rule_counts) > 0:
    plt.figure(figsize=(10,6))
    rule_counts.sort_values(ascending=False).plot(kind='bar')
    plt.title("Violations per SHACL Rule (Corrected)")
    plt.ylabel("Number of violations")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("violation_per_rule_corrected.png", dpi=150)
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
    f.write("Violation counts per SHACL rule (corrected):\n")
    f.write(rule_counts.to_string())
    f.write("\n\nComparison of valid vs invalid:\n")
    f.write(summary_stats.to_string())
    f.write(f"\n\nCorrelation distance-cost: {corr_dist_cost:.3f}\n")
    f.write("\nMann-Whitney tests:\n")
    for metric in ['distance', 'cost']:
        _, p = stats.mannwhitneyu(valid[metric], invalid[metric], alternative='two-sided')
        f.write(f"{metric}: p={p:.4f}\n")
print("Detailed report saved to analysis_paper_report.txt")