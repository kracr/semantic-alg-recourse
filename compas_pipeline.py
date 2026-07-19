"""
End‑to‑end COMPAS counterfactual pipeline – FULL FEATURES VERSION.
Includes all standard features: juv_other_count, days_b_screening_arrest,
length_of_stay, plus metadata (age_cat, decile_score, score_text).
"""

import uuid
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import dice_ml
import pyshacl
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, SH, XSD
import morph_kgc

warnings.filterwarnings("ignore")

# ============================================================
# 0. Load and preprocess COMPAS data (FULL features)
# ============================================================
def load_compas_data():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)

    # Keep all relevant columns (including new ones)
    keep_cols = [
        "age", "age_cat", "sex", "race", "priors_count",
        "c_charge_degree", "juv_fel_count", "juv_misd_count", "juv_other_count",
        "days_b_screening_arrest", "decile_score", "score_text",
        "c_jail_in", "c_jail_out", "is_recid", "two_year_recid"
    ]
    df = df[keep_cols]

    # ---- Standard pre-processing (aligns with mlr3 / ProPublica) ----
    # 1. Remove outliers for days_b_screening_arrest (|value| >= 30)
    df = df[df['days_b_screening_arrest'].abs() <= 30]

    # 2. Keep only valid is_recid (not -1)
    df = df[df['is_recid'] != -1]

    # 3. Keep only F or M charge degree
    df = df[df['c_charge_degree'].isin(['F', 'M'])]

    # 4. Remove rows where score_text is 'N/A'
    df = df[df['score_text'] != 'N/A']

    # 5. Compute length_of_stay (days in jail)
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df = df.dropna(subset=['length_of_stay'])

    # 6. Drop original jail dates (already used)
    df = df.drop(columns=['c_jail_in', 'c_jail_out', 'is_recid'])

    # Rename target
    df = df.rename(columns={"two_year_recid": "recidivism"})
    df['recidivism'] = df['recidivism'].astype(int)

    return df

print("Loading and preprocessing COMPAS data (full features)...")
df = load_compas_data()

# ---- Feature definitions ----
# Continuous features (used in model)
CONTINUOUS = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "days_b_screening_arrest", "length_of_stay"
]

# Categorical features (used in model)
CATEGORICAL = ["sex", "race", "c_charge_degree"]

# Metadata features (not used in model, but stored in RDF)
METADATA = ["age_cat", "decile_score", "score_text"]

TARGET = "recidivism"

# All features (model inputs) – note: metadata excluded from model
ALL_FEATURES = CONTINUOUS + CATEGORICAL

X = df[ALL_FEATURES]
y = df[TARGET]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

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
rejected_indices = np.where(y_pred == 1)[0]
num_to_evaluate = min(100, len(rejected_indices))
selected_indices = rejected_indices[:num_to_evaluate]
print(f"Evaluating {num_to_evaluate} recidivism-predicted defendants.")

# ============================================================
# 1. Setup DiCE
# ============================================================
class EncodedModel:
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_enc = self.preprocessor.transform(X)
        return self.model.predict(X_enc)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_enc = self.preprocessor.transform(X)
        return self.model.predict_proba(X_enc)

wrapped_model = EncodedModel(clf, preprocessor, ALL_FEATURES)

train_df = pd.concat([X_train_raw, y_train], axis=1)
dice_data = dice_ml.Data(
    dataframe=train_df,
    continuous_features=CONTINUOUS,
    outcome_name=TARGET,
)
dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn")

# ============================================================
# 2. Immutability, cost, ordinal maps
# ============================================================
# Immutable features: protected attributes + fixed historical facts
IMMUTABLE = ["sex", "race"]
ALL_VARY_FEATURES = [f for f in ALL_FEATURES if f not in IMMUTABLE]

USER_COST = {
    'age': 1.0,
    'priors_count': 5.0,
    'c_charge_degree': 3.0,
    'days_b_screening_arrest': 2.0,
    'length_of_stay': 2.0,
    'juv_fel_count': 4.0,        # added
    'juv_misd_count': 4.0,       # added
    'juv_other_count': 4.0,      # added
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
# 3. Generate counterfactuals
# ============================================================
print("\nGenerating counterfactuals with DiCE...")
all_candidates = []

for idx in tqdm(selected_indices):
    orig_row = X_test_raw.iloc[idx]
    if clf.predict(X_test_enc[idx].reshape(1, -1))[0] != 1:
        continue

    try:
        exp = dice_ml.Dice(dice_data, dice_model, method="genetic")
        cf = exp.generate_counterfactuals(
            orig_row.to_frame().T,
            total_CFs=5,
            desired_class="opposite",
            features_to_vary=ALL_VARY_FEATURES,
            # Permitted ranges (aligned with data & SHACL)
            permitted_range={
                "age": [18, 96],
                "priors_count": [0, 38],
                "days_b_screening_arrest": [-30, 30],
                "length_of_stay": [0, 365],
                "c_charge_degree": ["F", "M"]
            },
            diversity_weight=1.0,
            posthoc_sparsity_algorithm="binary",
            posthoc_sparsity_param=0.2,
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is not None and len(cf_df) > 0:
            for _, cf_row in cf_df.iterrows():
                full_row = orig_row.copy()
                for col in cf_row.index:
                    if col in full_row.index:
                        full_row[col] = cf_row[col]
                dist, cost = compute_distance_cost(orig_row, full_row)
                cf_id = f"cf_{idx}_{uuid.uuid4().hex[:6]}"
                all_candidates.append((idx, orig_row, full_row, dist, cost, cf_id))
    except Exception as e:
        print(f"Error for {idx}: {e}")

print(f"Total candidates: {len(all_candidates)}")
if not all_candidates:
    print("No CFs generated. Exiting.")
    raise SystemExit(0)

# ============================================================
# 4. Build DataFrame for RDF
# ============================================================
rows = []
applicants_with_cf = set(idx for idx, _, _, _, _, _ in all_candidates)

# ---- FIX: get metadata using label-based indexing ----
def get_metadata(pos_idx):
    orig_idx = X_test_raw.index[pos_idx]
    return df.loc[orig_idx][METADATA].to_dict()

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
    cf_pred = wrapped_model.predict(pd.DataFrame([cf_row[ALL_FEATURES]]))[0]
    d['predicted_recidivism'] = bool(cf_pred)
    rows.append(d)

df_rdf = pd.DataFrame(rows)
print(f"DataFrame for RDF has {len(df_rdf)} rows.")

# ============================================================
# 5. Morph‑KGC
# ============================================================
config_str = """
[CFSource]
mappings: compas_mapping_lean.yml
"""

print("\nRunning Morph-KGC...")
g = morph_kgc.materialize(config_str, python_source={"counterfactual_row": df_rdf})
print(f"Generated {len(g)} triples.")

# ============================================================
# 6. Load Ontology and SHACL, then validate
# ============================================================
onto = Graph()
try:
    onto.parse("compas_ontology_lean.ttl", format="turtle")
    g += onto
    print("Loaded compas_ontology.ttl")
except FileNotFoundError:
    print("Warning: compas_ontology.ttl not found.")

shapes = Graph()
try:
    shapes.parse("compas_shacl_lean.ttl", format="turtle")
    print("Loaded compas_shacl.ttl")
except FileNotFoundError:
    print("Warning: compas_shacl.ttl not found.")

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
# 7. Extract violations per CF
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
df_all.to_csv("compas_validation_table.csv", index=False)
print(f"Saved validation table ({len(df_all)} rows)")

# ============================================================
# 8. Add CounterfactualExplanation triples (metadata)
# ============================================================
def add_explanation_triples(graph, df_records, method_name="dice"):
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

g = add_explanation_triples(g, df_all, method_name="dice")
g.serialize("compas_counterfactuals.ttl", format="turtle")
print("Saved RDF with explanation triples.")

# ============================================================
# 9. Summary
# ============================================================
summary = df_all.groupby('applicant_idx').agg(
    num_cfs=('cf_id', 'count'),
    num_valid=('conforms', 'sum'),
    min_warning=('warning_count', 'min'),
    avg_distance=('distance', 'mean'),
    avg_cost=('cost', 'mean')
).reset_index()
summary.to_csv("compas_summary.csv", index=False)
print("Saved summary.")
print("\nAll done. Files: compas_validation_table.csv, compas_counterfactuals.ttl, compas_summary.csv")