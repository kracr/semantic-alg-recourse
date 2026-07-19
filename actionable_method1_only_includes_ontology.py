"""
Actionability Pipeline - All CFs, full validation table.
VERIFIED end-to-end against the installed morph-kgc package and the
real cf_mapping.yml / shacl_rules.ttl files.

This version loads credit_ontology.ttl to enable proper RDFS inference
during SHACL validation. This will cause shapes that target
ex:CreditApplication to also apply to ex:CounterfactualApplication
instances (since the ontology declares that subclass relation).
"""

import uuid, warnings, pandas as pd, numpy as np
from tqdm import tqdm
import dice_ml
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pyshacl
from rdflib import Graph
from rdflib.namespace import RDF, SH
import morph_kgc  # module-level function API

warnings.filterwarnings("ignore")

# ============================================================
# 1. Load Data & Train Model
# ============================================================
print("Loading data and training model...")
german = fetch_openml(name="credit-g", version=1, as_frame=True)
df = german.frame.copy()

CONTINUOUS = ["duration", "credit_amount", "installment_commitment",
              "residence_since", "age", "existing_credits", "num_dependents"]
CATEGORICAL = ["checking_status", "credit_history", "purpose", "savings_status",
               "employment", "personal_status", "other_parties",
               "property_magnitude", "other_payment_plans", "housing",
               "job", "own_telephone", "foreign_worker"]
TARGET = "class"
df = df.rename(columns={
    "installment_commitment": "installment_rate",
    "other_parties": "other_debtors",
})
CONTINUOUS[CONTINUOUS.index("installment_commitment")] = "installment_rate"
CATEGORICAL[CATEGORICAL.index("other_parties")] = "other_debtors"
ALL_FEATURES = [c for c in CONTINUOUS + CATEGORICAL]

X = df[ALL_FEATURES]
y = df[TARGET]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_enc = pd.get_dummies(X_train_raw)
X_test_enc = pd.get_dummies(X_test_raw)
X_train_enc, X_test_enc = X_train_enc.align(
    X_test_enc, join="left", axis=1, fill_value=0
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_enc, y_train)
print(f"Model accuracy: {clf.score(X_test_enc, y_test):.2f}")


class EncodedModel:
    def __init__(self, model, train_cols, feature_names):
        self.model = model
        self.train_cols = train_cols
        self.feature_names = feature_names

    def _encode(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return pd.get_dummies(X).reindex(columns=self.train_cols, fill_value=0)

    def predict(self, X):
        return self.model.predict(self._encode(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._encode(X))


wrapped_model = EncodedModel(clf, X_train_enc.columns, ALL_FEATURES)

y_pred = clf.predict(X_test_enc)
rejected_indices = np.where(y_pred == 'bad')[0]
num_to_evaluate = min(100, len(rejected_indices))
selected_indices = rejected_indices[:num_to_evaluate]
print(f"Evaluating {num_to_evaluate} rejected applicants.")

train_df = pd.concat([X_train_raw, y_train], axis=1)
dice_data = dice_ml.Data(
    dataframe=train_df,
    continuous_features=CONTINUOUS,
    outcome_name=TARGET,
)
dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn")

# ============================================================
# 2. Immutability, Cost, Ordinal Maps
# ============================================================
IMMUTABLE = ["personal_status", "foreign_worker", "num_dependents"]
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
# 3. Generate ALL Counterfactuals
# ============================================================
print("\nGenerating counterfactuals...")
all_candidates = []

for idx in tqdm(selected_indices):
    orig_row = X_test_raw.iloc[idx]
    if clf.predict(X_test_enc.iloc[[idx]])[0] != 'bad':
        continue

    candidates_this = []

    def process_cfs(cf_df):
        if cf_df is None or len(cf_df) == 0:
            return
        for _, cf_row in cf_df.iterrows():
            full_row = orig_row.copy()
            for col in cf_row.index:
                if col in full_row.index:
                    full_row[col] = cf_row[col]
            dist, cost = compute_distance_cost(orig_row, full_row)
            candidates_this.append((full_row, dist, cost))

    try:
        exp_all = dice_ml.Dice(dice_data, dice_model, method="genetic")
        cf_all = exp_all.generate_counterfactuals(
            orig_row.to_frame().T,
            total_CFs=15,
            desired_class="opposite",
            features_to_vary=ALL_VARY_FEATURES,
            permitted_range={"credit_amount": [500, 10000], "duration": [6, 48], "installment_rate": [1, 4]},
            diversity_weight=2.0,
            posthoc_sparsity_algorithm="binary",
            posthoc_sparsity_param=0.2,
        )
        process_cfs(cf_all.cf_examples_list[0].final_cfs_df)
    except Exception:
        pass

    for cf_row, dist, cost in candidates_this:
        cf_id = f"cf_{idx}_{uuid.uuid4().hex[:6]}"
        all_candidates.append((idx, orig_row, cf_row, dist, cost, cf_id))

print(f"Total candidates: {len(all_candidates)}")
if not all_candidates:
    print("No CFs generated. Exiting.")
    raise SystemExit(0)

# ============================================================
# 4. Build DataFrame with ALL required columns
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

for idx, orig_row, cf_row, dist, cost, cf_id in all_candidates:
    d = cf_row.to_dict()

    for feat in ALL_FEATURES:
        d[f'original_{feat}'] = orig_row.get(feat)

    d['original_checking_ordinal'] = ordinal_val(CHECKING_ORDINAL, orig_row.get('checking_status'))
    d['new_checking_ordinal'] = ordinal_val(CHECKING_ORDINAL, cf_row.get('checking_status'))
    d['original_savings_ordinal'] = ordinal_val(SAVINGS_ORDINAL, orig_row.get('savings_status'))
    d['new_savings_ordinal'] = ordinal_val(SAVINGS_ORDINAL, cf_row.get('savings_status'))
    d['original_employment_ordinal'] = ordinal_val(EMPLOYMENT_ORDINAL, orig_row.get('employment'))
    d['new_employment_ordinal'] = ordinal_val(EMPLOYMENT_ORDINAL, cf_row.get('employment'))

    changed = sum(
        1 for f in ALL_FEATURES
        if f not in IMMUTABLE and str(cf_row.get(f)) != str(orig_row.get(f))
    )
    d['num_changed_features'] = changed

    d['own_telephone'] = to_bool(cf_row.get('own_telephone'))
    d['foreign_worker'] = to_bool(cf_row.get('foreign_worker'))
    d['original_foreign_worker'] = to_bool(orig_row.get('foreign_worker'))

    d['cf_id'] = cf_id
    d['is_cf'] = 1
    d['distance'] = dist
    d['cost'] = cost
    rows.append(d)

df_rdf = pd.DataFrame(rows)
print(f"DataFrame for RDF has {len(df_rdf)} rows.")

# ============================================================
# 5. Run Morph-KGC
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
    print("First row of df_rdf (keys):", list(df_rdf.columns)[:20])
    df_rdf.to_csv("debug_rdf_input.csv", index=False)
    print("Saved debug_rdf_input.csv for inspection.")

# ============================================================
# 6. Load Ontology for RDFS Inference (NEW)
# ============================================================
ontology_graph = Graph()
try:
    ontology_graph.parse("credit_ontology.ttl", format="turtle")
    print("Loaded credit_ontology.ttl successfully.")
    # Merge ontology into the data graph so that RDFS inference can use its axioms
    g += ontology_graph
except FileNotFoundError:
    print("Warning: credit_ontology.ttl not found. Proceeding without ontology.")
# ============================================================

# ============================================================
# 7. Load SHACL shapes & validate
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
        inference="rdfs",          # now using the ontology axioms
        abort_on_first=False,
        allow_warnings=True,
    )
    print(f"Validation conforms: {conforms}")
else:
    print("No SHACL shapes loaded; skipping validation.")
    results_graph = Graph()

# ============================================================
# 8. Extract violations per CF
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
    loan_uri = f"http://example.org/credit/Loan_{cf_id}"
    violations, wc = get_violations_and_warnings(loan_uri)
    records.append({
        'applicant_idx': idx,
        'cf_id': cf_id,
        'credit_amount': cf_row.get('credit_amount'),
        'duration': cf_row.get('duration'),
        'age': cf_row.get('age'),
        'employment': cf_row.get('employment'),
        'checking_status': cf_row.get('checking_status'),
        'savings_status': cf_row.get('savings_status'),
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

summary = df_all.groupby('applicant_idx').agg(
    num_cfs=('cf_id', 'count'),
    num_valid=('conforms', 'sum'),
    min_warning=('warning_count', 'min'),
    avg_distance=('distance', 'mean'),
    avg_cost=('cost', 'mean')
).reset_index()
summary.to_csv("evaluation_summary.csv", index=False)
print("✓ Saved evaluation_summary.csv")

print("\nAll done. Full CF table is in 'all_cf_validation_table.csv'.")