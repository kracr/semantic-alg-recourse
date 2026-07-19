"""
Actionability Pipeline – Action‑Rules (2025)

Generates counterfactuals by mining action rules,
converts bin recommendations back to numeric values,
then RDF, SHACL, validation, and summary.
"""

import uuid
import warnings
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pyshacl
from rdflib import Graph
from rdflib.namespace import RDF, SH
import morph_kgc

from action_rules import ActionRules

warnings.filterwarnings("ignore")

# ============================================================
# 1. Load Data & Train Model (RandomForest)
# ============================================================
print("Loading data and training model...")
german = fetch_openml(name="credit-g", version=1, as_frame=True)
df = german.frame.copy()

CONTINUOUS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents"
]
CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties",
    "property_magnitude", "other_payment_plans", "housing",
    "job", "own_telephone", "foreign_worker"
]
TARGET = "class"

# Rename to match the original mapping
df = df.rename(columns={
    "installment_commitment": "installment_rate",
    "other_parties": "other_debtors",
})
CONTINUOUS[CONTINUOUS.index("installment_commitment")] = "installment_rate"
CATEGORICAL[CATEGORICAL.index("other_parties")] = "other_debtors"
ALL_FEATURES = CONTINUOUS + CATEGORICAL

X = df[ALL_FEATURES]
y = df[TARGET]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Encode categoricals for the model (one‑hot)
X_train_enc = pd.get_dummies(X_train_raw)
X_test_enc = pd.get_dummies(X_test_raw)
X_train_enc, X_test_enc = X_train_enc.align(
    X_test_enc, join="left", axis=1, fill_value=0
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_enc, y_train)
print(f"Model accuracy: {clf.score(X_test_enc, y_test):.2f}")

y_pred = clf.predict(X_test_enc)
rejected_indices = np.where(y_pred == 'bad')[0]
num_to_evaluate = min(100, len(rejected_indices))
selected_indices = rejected_indices[:num_to_evaluate]
print(f"Evaluating {num_to_evaluate} rejected applicants.")

# ============================================================
# 2. Set up Action‑Rules (v2.x API)
# ============================================================
# Binning continuous features
N_BINS = 4
bin_edges = {}
train_df_binned = X_train_raw.copy()
for c in CONTINUOUS:
    train_df_binned[c], edges = pd.qcut(X_train_raw[c], q=N_BINS, duplicates="drop", retbins=True)
    train_df_binned[c] = train_df_binned[c].astype(str)
    bin_edges[c] = edges

def bin_value(feat, value):
    edges = bin_edges[feat]
    clipped = min(max(value, edges[0]), edges[-1])
    interval = pd.cut([clipped], bins=edges, include_lowest=True)[0]
    return str(interval)

# Compute bin midpoints
bin_midpoints = {}
for c in CONTINUOUS:
    edges = bin_edges[c]
    intervals = pd.IntervalIndex.from_breaks(edges, closed='right')
    midpoint_map = {str(interval): (interval.left + interval.right) / 2 for interval in intervals}
    bin_midpoints[c] = midpoint_map

# Prepare training DataFrame with binned continuous + categorical strings + target
train_df_final = train_df_binned.copy()
for cat in CATEGORICAL:
    train_df_final[cat] = X_train_raw[cat].astype(str)
# ADD TARGET COLUMN
train_df_final[TARGET] = y_train.values

stable_attributes = ["personal_status", "foreign_worker", "num_dependents"]
flexible_attributes = [f for f in ALL_FEATURES if f not in stable_attributes]

min_support_count = max(5, int(0.02 * len(train_df_final)))

ar = ActionRules(
    min_stable_attributes=1,
    min_flexible_attributes=1,
    min_undesired_support=min_support_count,
    min_undesired_confidence=0.6,
    min_desired_support=min_support_count,
    min_desired_confidence=0.6,
    verbose=False,
)

ar.fit(
    data=train_df_final,
    stable_attributes=stable_attributes,
    flexible_attributes=flexible_attributes,
    target=TARGET,
    target_undesired_state="bad",
    target_desired_state="good",
)

print(f"Mined {len(ar.get_rules().get_ar_notation())} action rules.")

# ============================================================
# 3. Immutability, Cost, Ordinal Maps (same as original)
# ============================================================
IMMUTABLE = stable_attributes
ALL_VARY_FEATURES = [f for f in ALL_FEATURES if f not in IMMUTABLE]

USER_COST = {
    'credit_amount': 2.0, 'duration': 2.0, 'installment_rate': 3.0,
    'own_telephone': 1.0, 'savings_status': 5.0, 'checking_status': 4.0,
    'employment': 5.0, 'existing_credits': 3.0, 'residence_since': 4.0,
    'housing': 3.0, 'job': 3.0, 'property_magnitude': 4.0,
    'other_debtors': 3.0, 'other_payment_plans': 3.0,
    'credit_history': 5.0, 'purpose': 2.0,
}
DEFAULT_COST = 1.0

EMPLOYMENT_ORDINAL = {
    "unemployed": 1, "unemp": 1,
    "<1": 2, "less than 1 year": 2,
    "1<=x<4": 3, "1<x<4": 3,
    "4<=x<7": 4, "4<x<7": 4,
    ">=7": 5, "7+": 5,
}
SAVINGS_ORDINAL = {
    "no known savings": 0, "unknown": 0,
    "<100": 1,
    "100<=x<500": 2,
    "500<=x<1000": 3,
    ">=1000": 4,
}
CHECKING_ORDINAL = {
    "no checking": 0,
    "<0": 1,
    "0<=x<200": 2,
    ">=200": 3,
}

def ordinal_val(mapping, val):
    if pd.isna(val):
        return -1
    v = str(val).strip().lower()
    for key, rank in mapping.items():
        if key.lower() in v:
            return rank
    return -1

def compute_distance_cost(orig_row, full_row):
    dist, cost = 0.0, 0.0
    for feat in ALL_FEATURES:
        if feat in IMMUTABLE:
            continue
        old, new = orig_row.get(feat), full_row.get(feat)
        if pd.isna(old) or pd.isna(new):
            continue
        if feat in CONTINUOUS:
            try:
                if abs(float(old) - float(new)) > 1e-9:
                    dist += abs(float(new) - float(old))
                    cost += USER_COST.get(feat, DEFAULT_COST)
            except Exception:
                if str(old) != str(new):
                    dist += 1.0
                    cost += USER_COST.get(feat, DEFAULT_COST)
        else:
            if str(old) != str(new):
                dist += 1.0
                cost += USER_COST.get(feat, DEFAULT_COST)
    return dist, cost

def to_bool(v):
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in ('yes', 'true', '1', 'a192', 'a201')

# ============================================================
# 4. Generate Counterfactuals – with robust bin-to-numeric conversion
# ============================================================
print("\nGenerating counterfactuals with Action-Rules...")
all_candidates = []
applicant_cf_counts = {}

for idx in tqdm(selected_indices):
    orig_row = X_test_raw.iloc[idx]
    if clf.predict(X_test_enc.iloc[[idx]])[0] != 'bad':
        continue

    try:
        # Binarise the input row for the action_rules predictor
        row_for_predict = orig_row.copy()
        for feat in CONTINUOUS:
            row_for_predict[feat] = bin_value(feat, orig_row[feat])
        row_for_predict = row_for_predict.astype(str)

        cf_df = ar.predict(row_for_predict)
    except Exception as e:
        print(f"\nError generating CF for applicant {idx}: {e}")
        cf_df = pd.DataFrame()

    if cf_df.empty:
        applicant_cf_counts[idx] = 0
        continue

    unique_cfs = []
    seen = set()
    for _, predicted_row in cf_df.iterrows():
        cf_dict = orig_row.to_dict()
        for col in cf_df.columns:
            if ' (Recommended)' in col or '(Recommended)' in col:
                feat = col.replace(' (Recommended)', '').replace('(Recommended)', '')
                if feat in ALL_FEATURES:
                    rec_val = predicted_row[col]
                    if feat in CONTINUOUS:
                        # Map bin string to midpoint
                        if isinstance(rec_val, str) and rec_val in bin_midpoints[feat]:
                            rec_val = bin_midpoints[feat][rec_val]
                        else:
                            # Fallback: extract numbers
                            nums = re.findall(r"[-+]?\d*\.?\d+", str(rec_val))
                            if len(nums) >= 2:
                                rec_val = (float(nums[0]) + float(nums[1])) / 2
                            elif len(nums) == 1:
                                rec_val = float(nums[0])
                            else:
                                rec_val = orig_row[feat]
                        rec_val = float(rec_val)
                    cf_dict[feat] = rec_val
        key = tuple(cf_dict.get(f, None) for f in ALL_FEATURES)
        if key not in seen:
            seen.add(key)
            unique_cfs.append(cf_dict)

    applicant_cf_counts[idx] = len(unique_cfs)

    for cf_dict in unique_cfs[:15]:
        full_row = orig_row.copy()
        for col in ALL_FEATURES:
            if col in cf_dict:
                full_row[col] = cf_dict[col]
        dist, cost = compute_distance_cost(orig_row, full_row)
        cf_id = f"cf_{idx}_{uuid.uuid4().hex[:6]}"
        all_candidates.append((idx, orig_row, full_row, dist, cost, cf_id))

zero_cf_applicants = [idx for idx, count in applicant_cf_counts.items() if count == 0]
if zero_cf_applicants:
    print(f"\nApplicants with 0 CFs: {zero_cf_applicants} (total {len(zero_cf_applicants)})")
else:
    print("\nAll applicants generated at least one CF.")

print(f"Total candidates: {len(all_candidates)}")
if not all_candidates:
    print("No CFs generated for any applicant. Exiting.")
    raise SystemExit(0)

# ============================================================
# 5. Build DataFrame for RDF (numeric values for continuous)
# ============================================================
rows = []

applicants_with_cf = set(idx for idx, _, _, _, _, _ in all_candidates)
for idx in applicants_with_cf:
    orig = X_test_raw.iloc[idx]
    d = orig.to_dict()
    d['cf_id'] = f'original_{idx}'
    d['is_cf'] = 0
    d['original_checking_ordinal'] = ordinal_val(CHECKING_ORDINAL, orig.get('checking_status'))
    d['original_savings_ordinal'] = ordinal_val(SAVINGS_ORDINAL, orig.get('savings_status'))
    d['original_employment_ordinal'] = ordinal_val(EMPLOYMENT_ORDINAL, orig.get('employment'))
    d['own_telephone'] = to_bool(orig.get('own_telephone'))
    d['foreign_worker'] = to_bool(orig.get('foreign_worker'))
    rows.append(d)

for idx, orig_raw, cf_raw, dist, cost, cf_id in all_candidates:
    d = cf_raw.to_dict()
    for feat in ALL_FEATURES:
        d[f'original_{feat}'] = orig_raw.get(feat)

    d['original_checking_ordinal'] = ordinal_val(CHECKING_ORDINAL, orig_raw.get('checking_status'))
    d['new_checking_ordinal'] = ordinal_val(CHECKING_ORDINAL, cf_raw.get('checking_status'))
    d['original_savings_ordinal'] = ordinal_val(SAVINGS_ORDINAL, orig_raw.get('savings_status'))
    d['new_savings_ordinal'] = ordinal_val(SAVINGS_ORDINAL, cf_raw.get('savings_status'))
    d['original_employment_ordinal'] = ordinal_val(EMPLOYMENT_ORDINAL, orig_raw.get('employment'))
    d['new_employment_ordinal'] = ordinal_val(EMPLOYMENT_ORDINAL, cf_raw.get('employment'))

    changed = sum(
        1 for f in ALL_FEATURES
        if f not in IMMUTABLE and str(cf_raw.get(f)) != str(orig_raw.get(f))
    )
    d['num_changed_features'] = changed

    d['own_telephone'] = to_bool(cf_raw.get('own_telephone'))
    d['foreign_worker'] = to_bool(cf_raw.get('foreign_worker'))
    d['original_foreign_worker'] = to_bool(orig_raw.get('foreign_worker'))

    d['cf_id'] = cf_id
    d['is_cf'] = 1
    d['distance'] = dist
    d['cost'] = cost
    rows.append(d)

df_rdf = pd.DataFrame(rows)
print(f"DataFrame for RDF has {len(df_rdf)} rows.")

# ============================================================
# 6. Run Morph‑KGC
# ============================================================
config_str = """
[CFSource]
mappings: cf_mapping.yml
"""

print("\nRunning Morph-KGC...")
g = morph_kgc.materialize(config_str, python_source={"counterfactual_row": df_rdf})
print(f"Generated {len(g)} triples.")
if len(g) == 0:
    print("WARNING: No triples generated. Check mapping file and column names.")
    df_rdf.to_csv("debug_rdf_input.csv", index=False)
    print("Saved debug_rdf_input.csv for inspection.")

# ============================================================
# 7. SHACL Validation
# ============================================================
shapes = Graph()
for fname in ["shacl_rules.ttl"]:
    try:
        shapes.parse(fname, format="turtle")
        print(f"Loaded {fname}")
    except FileNotFoundError:
        print(f"Warning: {fname} not found, skipping.")

if len(shapes) > 0:
    conforms, results_graph, _ = pyshacl.validate(
        data_graph=g,
        shacl_graph=shapes,
        inference="rdfs",
        abort_on_first=False,
        allow_warnings=True,
    )
    print(f"Validation conforms: {conforms}")
else:
    print("No SHACL shapes loaded; skipping validation.")
    results_graph = Graph()

# ============================================================
# 8. Extract Violations per CF
# ============================================================
def get_violations_and_warnings(focus_uri):
    vio, warns = [], 0
    for res in results_graph.subjects(RDF.type, SH.ValidationResult):
        for _, _, fnode in results_graph.triples((res, SH.focusNode, None)):
            if str(fnode) == focus_uri:
                sev = None
                for _, _, s in results_graph.triples((res, SH.resultSeverity, None)):
                    sev = s
                    break
                if sev == SH.Violation:
                    msg = None
                    for _, _, m in results_graph.triples((res, SH.resultMessage, None)):
                        msg = str(m)
                        break
                    if msg:
                        vio.append(msg)
                elif sev == SH.Warning:
                    warns += 1
                break
    return vio, warns

records = []
for idx, orig_raw, cf_raw, dist, cost, cf_id in all_candidates:
    loan_uri = f"http://example.org/credit/Loan_{cf_id}"
    violations, wc = get_violations_and_warnings(loan_uri)
    records.append({
        'applicant_idx': idx,
        'cf_id': cf_id,
        'credit_amount': cf_raw.get('credit_amount'),
        'duration': cf_raw.get('duration'),
        'age': cf_raw.get('age'),
        'employment': cf_raw.get('employment'),
        'checking_status': cf_raw.get('checking_status'),
        'savings_status': cf_raw.get('savings_status'),
        'conforms': len(violations) == 0,
        'violations': "; ".join(violations) if violations else "No violation",
        'warning_count': wc,
        'distance': dist,
        'cost': cost,
    })

df_all = pd.DataFrame(records)
df_all.to_csv("all_cf_validation_table.csv", index=False)
print(f"\n✓ Saved all_cf_validation_table.csv ({len(df_all)} rows)")

g.serialize("counterfactuals_rdf.ttl", format="turtle")
print("✓ Saved counterfactuals_rdf.ttl")

# ============================================================
# 9. Summary (includes applicants with zero CFs)
# ============================================================
summary_list = []
for idx in selected_indices:
    count = applicant_cf_counts.get(idx, 0)
    if count > 0:
        sub = df_all[df_all['applicant_idx'] == idx]
        num_valid = sub['conforms'].sum()
        min_warning = sub['warning_count'].min()
        avg_dist = sub['distance'].mean()
        avg_cost = sub['cost'].mean()
    else:
        num_valid = 0
        min_warning = 0
        avg_dist = np.nan
        avg_cost = np.nan
    summary_list.append({
        'applicant_idx': idx,
        'num_cfs': count,
        'num_valid': num_valid,
        'min_warning': min_warning,
        'avg_distance': avg_dist,
        'avg_cost': avg_cost
    })

summary_df = pd.DataFrame(summary_list)
summary_df.to_csv("evaluation_summary.csv", index=False)
print("✓ Saved evaluation_summary.csv (includes all applicants, even those with 0 CFs)")

print("\nAll done. Full CF table is in 'all_cf_validation_table.csv'.")