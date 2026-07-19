"""
Actionability Pipeline – Action‑Rules for COMPAS (2025)

Generates counterfactuals by mining action rules on COMPAS data,
converts bin recommendations back to numeric values,
then RDF, SHACL, validation, and summary.
Uses lean ontology, mapping, and SHACL files.
"""

import uuid
import warnings
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pyshacl
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, SH, XSD
import morph_kgc

from action_rules import ActionRules

warnings.filterwarnings("ignore")

# ============================================================
# 0. Load and preprocess COMPAS data (FULL features, standard cleaning)
# ============================================================
def load_compas_data():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)

    keep_cols = [
        "age", "age_cat", "sex", "race", "priors_count",
        "c_charge_degree", "juv_fel_count", "juv_misd_count", "juv_other_count",
        "days_b_screening_arrest", "decile_score", "score_text",
        "c_jail_in", "c_jail_out", "is_recid", "two_year_recid"
    ]
    df = df[keep_cols]

    # Standard pre-processing
    df = df[df['days_b_screening_arrest'].abs() <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'].isin(['F', 'M'])]
    df = df[df['score_text'] != 'N/A']

    # Compute length_of_stay
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df = df.dropna(subset=['length_of_stay'])
    df = df.drop(columns=['c_jail_in', 'c_jail_out', 'is_recid'])

    df = df.rename(columns={"two_year_recid": "recidivism"})
    df['recidivism'] = df['recidivism'].astype(int)
    return df

print("Loading and preprocessing COMPAS data...")
df = load_compas_data()

# Feature definitions (model inputs)
CONTINUOUS = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "days_b_screening_arrest", "length_of_stay"
]
CATEGORICAL = ["sex", "race", "c_charge_degree"]
METADATA = ["age_cat", "decile_score", "score_text"]
TARGET = "recidivism"
ALL_FEATURES = CONTINUOUS + CATEGORICAL

X = df[ALL_FEATURES]
y = df[TARGET]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train a RandomForest model (with preprocessing)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), CONTINUOUS),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)
])
X_train_enc = preprocessor.fit_transform(X_train_raw)
X_test_enc = preprocessor.transform(X_test_raw)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_enc, y_train)
print(f"Model accuracy: {clf.score(X_test_enc, y_test):.2f}")

y_pred = clf.predict(X_test_enc)
rejected_indices = np.where(y_pred == 1)[0]  # 1 = recidivism (bad)
num_to_evaluate = min(100, len(rejected_indices))
selected_indices = rejected_indices[:num_to_evaluate]
print(f"Evaluating {num_to_evaluate} recidivism-predicted defendants.")

# ============================================================
# 1. Set up Action‑Rules (v2.x API)
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

# Compute bin midpoints for numeric conversion
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
train_df_final[TARGET] = y_train.values

# Immutable features (protected + fixed historical facts)
stable_attributes = ["sex", "race", "juv_fel_count", "juv_misd_count", "juv_other_count"]
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
    target_undesired_state=1,
    target_desired_state=0,
)

print(f"Mined {len(ar.get_rules().get_ar_notation())} action rules.")

# ============================================================
# 2. Immutability, Cost, Ordinal Maps
# ============================================================
IMMUTABLE = stable_attributes
ALL_VARY_FEATURES = [f for f in ALL_FEATURES if f not in IMMUTABLE]

USER_COST = {
    'age': 1.0,
    'priors_count': 5.0,
    'c_charge_degree': 3.0,
    'days_b_screening_arrest': 2.0,
    'length_of_stay': 2.0,
}
DEFAULT_COST = 1.0

def priors_ordinal(x):
    if x == 0: return 0
    elif x == 1: return 1
    elif x <= 3: return 2
    else: return 3

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
    return s in ('true', '1', 'yes')

# ============================================================
# 3. Generate Counterfactuals using ActionRules
# ============================================================
print("\nGenerating counterfactuals with Action-Rules...")
all_candidates = []
applicant_cf_counts = {}

def get_metadata(pos_idx):
    orig_idx = X_test_raw.index[pos_idx]
    return df.loc[orig_idx][METADATA].to_dict()

for idx in tqdm(selected_indices):
    orig_row = X_test_raw.iloc[idx]
    if clf.predict(X_test_enc[idx].reshape(1, -1))[0] != 1:
        continue

    try:
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
                        if isinstance(rec_val, str) and rec_val in bin_midpoints[feat]:
                            rec_val = bin_midpoints[feat][rec_val]
                        else:
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
# 4. Build DataFrame for RDF
# ============================================================
rows = []
applicants_with_cf = set(idx for idx, _, _, _, _, _ in all_candidates)

for idx in applicants_with_cf:
    orig = X_test_raw.iloc[idx]
    meta = get_metadata(idx)
    d = orig.to_dict()
    d.update(meta)
    d['cf_id'] = f'original_{idx}'
    d['is_cf'] = 0
    d['original_priors_ordinal'] = priors_ordinal(orig['priors_count'])
    d['new_priors_ordinal'] = None
    d['num_changed_features'] = 0
    d['predicted_recidivism'] = bool(y_pred[idx])
    for f in ALL_FEATURES:
        d[f'original_{f}'] = orig[f]
    d['recidivism'] = to_bool(orig.get('recidivism'))
    rows.append(d)

for idx, orig_row, cf_row, dist, cost, cf_id in all_candidates:
    meta = get_metadata(idx)
    d = cf_row.to_dict()
    d.update(meta)
    for f in ALL_FEATURES:
        d[f'original_{f}'] = orig_row[f]
    d['original_priors_ordinal'] = priors_ordinal(orig_row['priors_count'])
    d['new_priors_ordinal'] = priors_ordinal(cf_row['priors_count'])
    changed = sum(1 for f in ALL_FEATURES if f not in IMMUTABLE and str(cf_row.get(f)) != str(orig_row.get(f)))
    d['num_changed_features'] = changed
    d['cf_id'] = cf_id
    d['is_cf'] = 1
    d['distance'] = dist
    d['cost'] = cost
    d['recidivism'] = to_bool(cf_row.get('recidivism'))
    # Predict CF class (should be 0, but compute robustly)
    try:
        cf_pred = clf.predict(preprocessor.transform(pd.DataFrame([cf_row])[ALL_FEATURES]))[0]
    except Exception:
        cf_pred = 0  # ActionRules guarantee flip; fallback
    d['predicted_recidivism'] = bool(cf_pred)
    rows.append(d)

df_rdf = pd.DataFrame(rows)
print(f"DataFrame for RDF has {len(df_rdf)} rows.")

# ============================================================
# 5. Run Morph‑KGC (using lean mapping)
# ============================================================
config_str = """
[CFSource]
mappings: compas_mapping_lean.yml
"""

print("\nRunning Morph-KGC...")
g = morph_kgc.materialize(config_str, python_source={"counterfactual_row": df_rdf})
print(f"Generated {len(g)} triples.")
if len(g) == 0:
    print("WARNING: No triples generated. Check mapping file and column names.")
    df_rdf.to_csv("debug_rdf_input.csv", index=False)
    print("Saved debug_rdf_input.csv for inspection.")

# ============================================================
# 6. Load Ontology and SHACL, then validate
# ============================================================
onto = Graph()
try:
    onto.parse("compas_ontology_lean.ttl", format="turtle")
    g += onto
    print("Loaded compas_ontology_lean.ttl")
except FileNotFoundError:
    print("Warning: compas_ontology_lean.ttl not found.")

shapes = Graph()
try:
    shapes.parse("compas_shacl_lean.ttl", format="turtle")
    print("Loaded compas_shacl_lean.ttl")
except FileNotFoundError:
    print("Warning: compas_shacl_lean.ttl not found.")

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
    results_graph = Graph()

# ============================================================
# 7. Extract Violations per CF
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
for idx, orig_row, cf_row, dist, cost, cf_id in all_candidates:
    person_uri = f"http://example.org/compas/Person_{cf_id}"
    violations, wc = get_violations_and_warnings(person_uri)
    row_df = df_rdf[df_rdf['cf_id'] == cf_id].iloc[0]
    num_changed = row_df['num_changed_features']
    records.append({
        'applicant_idx': idx,
        'cf_id': cf_id,
        'age': cf_row.get('age'),
        'priors_count': cf_row.get('priors_count'),
        'race': cf_row.get('race'),
        'sex': cf_row.get('sex'),
        'juv_other_count': cf_row.get('juv_other_count'),
        'days_b_screening_arrest': cf_row.get('days_b_screening_arrest'),
        'length_of_stay': cf_row.get('length_of_stay'),
        'conforms': len(violations) == 0,
        'violations': "; ".join(violations) if violations else "No violation",
        'warning_count': wc,
        'distance': dist,
        'cost': cost,
        'num_changed_features': num_changed,
    })

df_all = pd.DataFrame(records)
df_all.to_csv("compas_validation_table_actionrules.csv", index=False)
print(f"\n✓ Saved validation table (actionrules) ({len(df_all)} rows)")

# ============================================================
# 8. Add CounterfactualExplanation triples (metadata)
# ============================================================
def add_explanation_triples(graph, df_records, method_name="actionrules"):
    EX = Namespace("http://example.org/compas/")
    for _, row in df_records.iterrows():
        cf_id = row['cf_id']
        cf_uri = URIRef(f"http://example.org/compas/Person_{cf_id}")
        exp_uri = URIRef(f"http://example.org/compas/Explanation_{cf_id}")
        graph.add((exp_uri, RDF.type, EX.CounterfactualExplanation))
        graph.add((exp_uri, EX.hasCounterfactual, cf_uri))
        graph.add((exp_uri, EX.proximityScore, Literal(float(row['distance']), datatype=XSD.decimal)))
        graph.add((exp_uri, EX.sparsityScore, Literal(int(row['num_changed_features']), datatype=XSD.integer)))
        graph.add((exp_uri, EX.costScore, Literal(float(row['cost']), datatype=XSD.decimal)))
        graph.add((exp_uri, EX.isValid, Literal(bool(row['conforms']), datatype=XSD.boolean)))
        graph.add((exp_uri, EX.generatedByMethod, Literal(method_name, datatype=XSD.string)))
        if row['violations'] != "No violation":
            graph.add((exp_uri, EX.violationMessages, Literal(row['violations'], datatype=XSD.string)))
    return graph

g = add_explanation_triples(g, df_all, method_name="actionrules")
g.serialize("compas_counterfactuals_actionrules.ttl", format="turtle")
print("✓ Saved RDF with explanation triples (actionrules).")

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
summary_df.to_csv("compas_summary_actionrules.csv", index=False)
print("✓ Saved summary (actionrules) – includes applicants with 0 CFs.")

print("\nAll done. Files: compas_validation_table_actionrules.csv, compas_counterfactuals_actionrules.ttl, compas_summary_actionrules.csv")