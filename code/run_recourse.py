#!/usr/bin/env python3
"""
Unified Recourse Pipeline – runs DiCE or ActionRules on German Credit or COMPAS.
All dataset parameters from dataset/<name>/config.yaml.
Preprocessing is dataset‑specific:
- german-credit: dummy encoding (no scaling, no stratify)
- compas: StandardScaler + OneHotEncoder with stratify (matches original COMPAS)
"""

import os, sys, argparse, yaml, uuid, warnings, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pyshacl
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, SH, XSD
import morph_kgc
import joblib
import dice_ml
from action_rules import ActionRules

warnings.filterwarnings("ignore")

# =============================================================================
# Custom DummyEncoder (no scaling, just one‑hot encoding)
# =============================================================================

class DummyEncoder:
    """One‑hot encode categorical features, keep continuous as is."""
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.dummy_cols = None

    def fit(self, X):
        dummy_df = pd.get_dummies(X[self.categorical_cols])
        self.dummy_cols = dummy_df.columns.tolist()
        return self

    def transform(self, X):
        dummy = pd.get_dummies(X[self.categorical_cols])
        dummy = dummy.reindex(columns=self.dummy_cols, fill_value=0)
        cont_cols = [c for c in X.columns if c not in self.categorical_cols]
        cont = X[cont_cols].reset_index(drop=True)
        dummy = dummy.reset_index(drop=True)
        return pd.concat([cont, dummy], axis=1).values

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# =============================================================================
# Helper functions
# =============================================================================

def load_config(dataset_name):
    config_path = os.path.join("dataset", dataset_name, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_dataset(config):
    ds_name = config['dataset_name']
    src = config['source_type']
    if src == 'openml':
        data = fetch_openml(name=config['openml_name'],
                            version=config.get('openml_version', 1),
                            as_frame=True)
        df = data.frame.copy()
        if ds_name == 'german-credit':
            df = df.rename(columns={
                "installment_commitment": "installment_rate",
                "other_parties": "other_debtors",
            })
        X = df[config['features']['continuous'] + config['features']['categorical']]
        y = df[config['target']]
        return df, X, y
    elif src == 'url':
        df = pd.read_csv(config['url'])
        if ds_name == 'compas':
            df = preprocess_compas(df)
        X = df[config['features']['continuous'] + config['features']['categorical']]
        y = df[config['target']]
        return df, X, y
    else:
        raise ValueError(f"Unknown source_type: {src}")

def preprocess_compas(df):
    keep = ["age","age_cat","sex","race","priors_count","c_charge_degree",
            "juv_fel_count","juv_misd_count","juv_other_count",
            "days_b_screening_arrest","decile_score","score_text",
            "c_jail_in","c_jail_out","is_recid","two_year_recid"]
    df = df[keep]
    df = df[df['days_b_screening_arrest'].abs() <= 30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'].isin(['F','M'])]
    df = df[df['score_text'] != 'N/A']
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df = df.dropna(subset=['length_of_stay'])
    df = df.drop(columns=['c_jail_in','c_jail_out','is_recid'])
    df = df.rename(columns={"two_year_recid":"recidivism"})
    df['recidivism'] = df['recidivism'].astype(int)
    return df

def to_bool(v):
    if isinstance(v, bool): return v
    if pd.isna(v): return False
    s = str(v).strip().lower()
    return s in ('yes','true','1','a192','a201')

def ordinal_val(mapping, val):
    if pd.isna(val): return -1
    v = str(val).strip().lower()
    for key, rank in mapping.items():
        if key.lower() in v:
            return rank
    return -1

def priors_ordinal(x):
    if x == 0: return 0
    elif x == 1: return 1
    elif x <= 3: return 2
    else: return 3

def compute_distance_cost(orig, cf, all_feats, cont, immut, cost_w):
    dist, cost = 0.0, 0.0
    for f in all_feats:
        if f in immut: continue
        old, new = orig.get(f), cf.get(f)
        if pd.isna(old) or pd.isna(new): continue
        if f in cont:
            try:
                if abs(float(old)-float(new)) > 1e-9:
                    dist += abs(float(new)-float(old))
                    cost += cost_w.get(f, cost_w.get('default',1.0))
            except:
                if str(old) != str(new):
                    dist += 1.0
                    cost += cost_w.get(f, cost_w.get('default',1.0))
        else:
            if str(old) != str(new):
                dist += 1.0
                cost += cost_w.get(f, cost_w.get('default',1.0))
    return dist, cost

def get_metadata(df, idx, meta_cols):
    if meta_cols:
        return df.loc[idx][meta_cols].to_dict()
    return {}

# =============================================================================
# CF generation functions
# =============================================================================

def gen_cfs_dice(orig, clf, preproc, all_feats, cont, cat, immut,
                 target_bad, config, X_train_raw, y_train, dice_data, dice_model):
    if clf.predict(preproc.transform(pd.DataFrame([orig])[all_feats]))[0] != target_bad:
        return []
    candidates = []
    try:
        exp = dice_ml.Dice(dice_data, dice_model, method="genetic")
        cf = exp.generate_counterfactuals(
            orig.to_frame().T,
            total_CFs=15,
            desired_class="opposite",
            features_to_vary=[f for f in all_feats if f not in immut],
            permitted_range=config.get('dice_permitted_range', {}),
            diversity_weight=2.0,
            posthoc_sparsity_algorithm="binary",
            posthoc_sparsity_param=0.2,
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is not None and len(cf_df) > 0:
            for _, row in cf_df.iterrows():
                full = orig.copy()
                for col in row.index:
                    if col in full.index:
                        full[col] = row[col]
                d, c = compute_distance_cost(orig, full, all_feats, cont, immut, config['cost_weights'])
                candidates.append((full, d, c))
    except Exception as e:
        print(f"DiCE error: {e}")
    return candidates

def gen_cfs_actionrules(orig, clf, preproc, all_feats, cont, cat, immut,
                        target_bad, config, ar, bin_edges, bin_midpoints):
    if clf.predict(preproc.transform(pd.DataFrame([orig])[all_feats]))[0] != target_bad:
        return []
    def bin_value(feat, value):
        edges = bin_edges[feat]
        clipped = min(max(value, edges[0]), edges[-1])
        interval = pd.cut([clipped], bins=edges, include_lowest=True)[0]
        return str(interval)
    candidates = []
    try:
        row_pred = orig.copy()
        for f in cont:
            row_pred[f] = bin_value(f, orig[f])
        row_pred = row_pred.astype(str)
        cf_df = ar.predict(row_pred)
        if cf_df.empty:
            return []
        unique = []
        seen = set()
        for _, pred_row in cf_df.iterrows():
            cf_dict = orig.to_dict()
            for col in cf_df.columns:
                if ' (Recommended)' in col or '(Recommended)' in col:
                    feat = col.replace(' (Recommended)','').replace('(Recommended)','')
                    if feat in all_feats:
                        val = pred_row[col]
                        if feat in cont:
                            if isinstance(val,str) and val in bin_midpoints[feat]:
                                val = bin_midpoints[feat][val]
                            else:
                                nums = re.findall(r"[-+]?\d*\.?\d+", str(val))
                                if len(nums) >= 2:
                                    val = (float(nums[0])+float(nums[1]))/2
                                elif len(nums) == 1:
                                    val = float(nums[0])
                                else:
                                    val = orig[feat]
                            val = float(val)
                        cf_dict[feat] = val
            key = tuple(cf_dict.get(f,None) for f in all_feats)
            if key not in seen:
                seen.add(key)
                unique.append(cf_dict)
        for cf_dict in unique[:15]:
            full = orig.copy()
            for f in all_feats:
                if f in cf_dict:
                    full[f] = cf_dict[f]
            d, c = compute_distance_cost(orig, full, all_feats, cont, immut, config['cost_weights'])
            candidates.append((full, d, c))
    except Exception as e:
        print(f"ActionRules error: {e}")
    return candidates

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["german-credit","compas"])
    parser.add_argument("--method", default="actionrules", choices=["dice","actionrules"])
    parser.add_argument("--num_cfs", type=int, default=100)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    config = load_config(args.dataset)
    ds_name = config['dataset_name']
    dataset_root = os.path.join("dataset", ds_name)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load data
    print(f"Loading {ds_name}...")
    df_raw, X, y = load_dataset(config)
    all_feats = config['features']['continuous'] + config['features']['categorical']
    cont = config['features']['continuous']
    cat = config['features']['categorical']
    target = config['target']
    target_bad = config['target_undesired']
    target_good = config['target_desired']
    immut = config['immutable_attributes']

    # Preprocessing settings from config
    prep_cfg = config.get('preprocessing', {})
    prep_method = prep_cfg.get('method', 'dummy')   # default to dummy
    use_stratify = prep_cfg.get('stratify', False)

    # Split
    if use_stratify and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    # Model & preprocessor paths (suffix based on prep method)
    suffix = "standard" if prep_method == "standard" else "dummy"
    model_path = f"models/{ds_name}_rf_{suffix}.joblib"
    preproc_path = f"models/{ds_name}_preproc_{suffix}.joblib"

    # Load or train
    if not args.retrain and os.path.exists(model_path) and os.path.exists(preproc_path):
        print("Loading saved model and preprocessor...")
        clf = joblib.load(model_path)
        preproc = joblib.load(preproc_path)
    else:
        print("Training new model...")
        if prep_method == "standard":
            # COMPAS style: StandardScaler + OneHotEncoder
            preproc = ColumnTransformer([
                ('num', StandardScaler(), cont),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
            ])
            X_train_enc = preproc.fit_transform(X_train)
            X_test_enc = preproc.transform(X_test)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_enc, y_train)
        else:
            # German Credit style: DummyEncoder (no scaling)
            preproc = DummyEncoder(categorical_cols=cat)
            X_train_enc = preproc.fit_transform(X_train)
            X_test_enc = preproc.transform(X_test)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_enc, y_train)
        acc = clf.score(X_test_enc, y_test)
        print(f"Accuracy: {acc:.2f}")
        joblib.dump(clf, model_path)
        joblib.dump(preproc, preproc_path)

    # Predict on test set
    X_test_enc = preproc.transform(X_test)
    y_pred = clf.predict(X_test_enc)
    rejected = np.where(y_pred == target_bad)[0]
    selected = rejected[:min(args.num_cfs, len(rejected))]
    print(f"Evaluating {len(selected)} rejected applicants.")

    # Setup method-specific
    if args.method == "dice":
        class EncodedModel:
            def __init__(self, model, preproc, feats):
                self.model = model
                self.preproc = preproc
                self.feats = feats
            def predict(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=self.feats)
                return self.model.predict(self.preproc.transform(X))
            def predict_proba(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=self.feats)
                return self.model.predict_proba(self.preproc.transform(X))
        wrapped = EncodedModel(clf, preproc, all_feats)
        train_df = pd.concat([X_train, y_train], axis=1)
        dice_data = dice_ml.Data(dataframe=train_df, continuous_features=cont, outcome_name=target)
        dice_model = dice_ml.Model(model=wrapped, backend="sklearn")
        ar = None
        bin_edges = bin_midpoints = None
    else:
        # ActionRules setup (binning always on raw data)
        N_BINS = 4
        bin_edges = {}
        train_binned = X_train.copy()
        for c in cont:
            train_binned[c], edges = pd.qcut(X_train[c], q=N_BINS, duplicates="drop", retbins=True)
            train_binned[c] = train_binned[c].astype(str)
            bin_edges[c] = edges
        bin_midpoints = {}
        for c in cont:
            edges = bin_edges[c]
            intervals = pd.IntervalIndex.from_breaks(edges, closed='right')
            bin_midpoints[c] = {str(i): (i.left+i.right)/2 for i in intervals}
        train_ar = train_binned.copy()
        for c in cat:
            train_ar[c] = X_train[c].astype(str)
        train_ar[target] = y_train.values
        stable = config['action_rules'].get('stable_attributes', immut)
        flexible = config['action_rules'].get('flexible_attributes', [])
        if not flexible:
            flexible = [f for f in all_feats if f not in stable]
        min_sup = max(5, int(config['action_rules'].get('min_support',0.02)*len(train_ar)))
        min_conf = config['action_rules'].get('min_confidence',0.6)
        ar = ActionRules(
            min_stable_attributes=config['action_rules'].get('min_stable',1),
            min_flexible_attributes=config['action_rules'].get('min_flexible',1),
            min_undesired_support=min_sup,
            min_undesired_confidence=min_conf,
            min_desired_support=min_sup,
            min_desired_confidence=min_conf,
            verbose=False,
        )
        ar.fit(train_ar, stable, flexible, target, target_bad, target_good)
        print(f"Mined {len(ar.get_rules().get_ar_notation())} action rules.")
        dice_data = dice_model = None

    # Generate CFs
    print(f"\nGenerating CFs with {args.method}...")
    all_candidates = []
    applicant_cf_counts = {}
    meta_cols = ["age_cat","decile_score","score_text"] if ds_name=="compas" else []

    for idx in tqdm(selected):
        orig = X_test.iloc[idx]
        if args.method == "dice":
            cfs = gen_cfs_dice(orig, clf, preproc, all_feats, cont, cat, immut,
                               target_bad, config, X_train, y_train, dice_data, dice_model)
        else:
            cfs = gen_cfs_actionrules(orig, clf, preproc, all_feats, cont, cat, immut,
                                      target_bad, config, ar, bin_edges, bin_midpoints)
        if not cfs:
            applicant_cf_counts[idx] = 0
            continue
        applicant_cf_counts[idx] = len(cfs)
        for cf_row, dist, cost in cfs[:15]:
            cf_id = f"cf_{idx}_{uuid.uuid4().hex[:6]}"
            all_candidates.append((idx, orig, cf_row, dist, cost, cf_id))

    print(f"Total candidates: {len(all_candidates)}")
    if not all_candidates:
        print("No CFs generated. Exiting.")
        sys.exit(0)

    # Build RDF DataFrame
    rows = []
    app_with_cf = {x[0] for x in all_candidates}
    for idx in app_with_cf:
        orig = X_test.iloc[idx]
        d = orig.to_dict()
        d.update(get_metadata(df_raw, X_test.index[idx], meta_cols))
        d['cf_id'] = f'original_{idx}'
        d['is_cf'] = 0
        if ds_name == 'german-credit':
            ord_maps = config.get('ordinal_mappings', {})
            d['original_checking_ordinal'] = ordinal_val(ord_maps.get('checking_status',{}), orig.get('checking_status'))
            d['original_savings_ordinal'] = ordinal_val(ord_maps.get('savings_status',{}), orig.get('savings_status'))
            d['original_employment_ordinal'] = ordinal_val(ord_maps.get('employment',{}), orig.get('employment'))
            d['own_telephone'] = to_bool(orig.get('own_telephone'))
            d['foreign_worker'] = to_bool(orig.get('foreign_worker'))
        elif ds_name == 'compas':
            d['original_priors_ordinal'] = priors_ordinal(orig['priors_count'])
            d['recidivism'] = to_bool(orig.get('recidivism'))
        for f in all_feats:
            d[f'original_{f}'] = orig[f]
        d['num_changed_features'] = 0
        rows.append(d)

    for idx, orig, cf, dist, cost, cf_id in all_candidates:
        d = cf.to_dict()
        d.update(get_metadata(df_raw, X_test.index[idx], meta_cols))
        for f in all_feats:
            d[f'original_{f}'] = orig[f]
        if ds_name == 'german-credit':
            ord_maps = config.get('ordinal_mappings', {})
            d['original_checking_ordinal'] = ordinal_val(ord_maps.get('checking_status',{}), orig.get('checking_status'))
            d['new_checking_ordinal'] = ordinal_val(ord_maps.get('checking_status',{}), cf.get('checking_status'))
            d['original_savings_ordinal'] = ordinal_val(ord_maps.get('savings_status',{}), orig.get('savings_status'))
            d['new_savings_ordinal'] = ordinal_val(ord_maps.get('savings_status',{}), cf.get('savings_status'))
            d['original_employment_ordinal'] = ordinal_val(ord_maps.get('employment',{}), orig.get('employment'))
            d['new_employment_ordinal'] = ordinal_val(ord_maps.get('employment',{}), cf.get('employment'))
            d['own_telephone'] = to_bool(cf.get('own_telephone'))
            d['foreign_worker'] = to_bool(cf.get('foreign_worker'))
            d['original_foreign_worker'] = to_bool(orig.get('foreign_worker'))
        elif ds_name == 'compas':
            d['original_priors_ordinal'] = priors_ordinal(orig['priors_count'])
            d['new_priors_ordinal'] = priors_ordinal(cf['priors_count'])
            d['recidivism'] = to_bool(cf.get('recidivism'))
            try:
                cf_pred = clf.predict(preproc.transform(pd.DataFrame([cf])[all_feats]))[0]
            except:
                cf_pred = 0
            d['predicted_recidivism'] = bool(cf_pred)
        changed = sum(1 for f in all_feats if f not in immut and str(cf.get(f)) != str(orig.get(f)))
        d['num_changed_features'] = changed
        d['cf_id'] = cf_id
        d['is_cf'] = 1
        d['distance'] = dist
        d['cost'] = cost
        rows.append(d)

    df_rdf = pd.DataFrame(rows)
    print(f"RDF DataFrame has {len(df_rdf)} rows.")

    # Morph-KGC
    rml = os.path.join(dataset_root, config['artifact_paths']['rml'])
    config_morph = f"[CFSource]\nmappings: {rml}"
    print("\nMorph-KGC...")
    g = morph_kgc.materialize(config_morph, python_source={"counterfactual_row": df_rdf})
    print(f"Generated {len(g)} triples.")
    if len(g) == 0:
        df_rdf.to_csv(f"results/{ds_name}_debug.csv", index=False)

    # Ontology & SHACL
    onto_path = os.path.join(dataset_root, config['artifact_paths']['ontology'])
    shacl_path = os.path.join(dataset_root, config['artifact_paths']['shacl'])
    onto = Graph()
    if os.path.exists(onto_path):
        onto.parse(onto_path, format="turtle")
        g += onto
        print(f"Loaded ontology from {onto_path}")
    shapes = Graph()
    if os.path.exists(shacl_path):
        shapes.parse(shacl_path, format="turtle")
        print(f"Loaded SHACL from {shacl_path}")
    if len(shapes) > 0:
        conforms, res_graph, _ = pyshacl.validate(g, shacl_graph=shapes, inference="rdfs",
                                                   abort_on_first=False, allow_warnings=True)
        print(f"Conforms: {conforms}")
    else:
        res_graph = Graph()

    # Extract violations
    def get_violations(focus_uri):
        vio, warns = [], 0
        for res in res_graph.subjects(RDF.type, SH.ValidationResult):
            for _, _, fnode in res_graph.triples((res, SH.focusNode, None)):
                if str(fnode) == focus_uri:
                    sev = None
                    for _, _, s in res_graph.triples((res, SH.resultSeverity, None)):
                        sev = s; break
                    if sev == SH.Violation:
                        msg = None
                        for _, _, m in res_graph.triples((res, SH.resultMessage, None)):
                            msg = str(m); break
                        if msg: vio.append(msg)
                    elif sev == SH.Warning:
                        warns += 1
                    break
        return vio, warns

    records = []
    for idx, orig, cf, dist, cost, cf_id in all_candidates:
        focus = f"http://example.org/credit/Loan_{cf_id}" if ds_name=='german-credit' else f"http://example.org/compas/Person_{cf_id}"
        vio, wc = get_violations(focus)
        row_info = df_rdf[df_rdf['cf_id']==cf_id]
        num_ch = row_info.iloc[0]['num_changed_features'] if not row_info.empty else 0
        rec = {'applicant_idx': idx, 'cf_id': cf_id, 'conforms': len(vio)==0,
               'violations': "; ".join(vio) if vio else "No violation",
               'warning_count': wc, 'distance': dist, 'cost': cost,
               'num_changed_features': num_ch}
        for f in ['age','credit_amount','priors_count','duration']:
            if f in cf:
                rec[f] = cf.get(f)
        records.append(rec)
    df_all = pd.DataFrame(records)
    out_csv = f"results/{ds_name}_{args.method}_validation.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved validation to {out_csv}")

    # Explanation triples
    EX = Namespace("http://example.org/explanation/")
    for _, row in df_all.iterrows():
        cf_id = row['cf_id']
        cf_uri = URIRef(f"http://example.org/credit/Loan_{cf_id}" if ds_name=='german-credit' else f"http://example.org/compas/Person_{cf_id}")
        exp_uri = URIRef(f"http://example.org/explanation/Explanation_{cf_id}")
        g.add((exp_uri, RDF.type, EX.CounterfactualExplanation))
        g.add((exp_uri, EX.hasCounterfactual, cf_uri))
        g.add((exp_uri, EX.proximityScore, Literal(float(row['distance']), datatype=XSD.decimal)))
        g.add((exp_uri, EX.sparsityScore, Literal(int(row['num_changed_features']), datatype=XSD.integer)))
        g.add((exp_uri, EX.costScore, Literal(float(row['cost']), datatype=XSD.decimal)))
        g.add((exp_uri, EX.isValid, Literal(bool(row['conforms']), datatype=XSD.boolean)))
        g.add((exp_uri, EX.generatedByMethod, Literal(args.method, datatype=XSD.string)))
        if row['violations'] != "No violation":
            g.add((exp_uri, EX.violationMessages, Literal(row['violations'], datatype=XSD.string)))
    g.serialize(f"results/{ds_name}_{args.method}_counterfactuals.ttl", format="turtle")

    # Summary
    summary = []
    for idx in selected:
        cnt = applicant_cf_counts.get(idx,0)
        if cnt > 0:
            sub = df_all[df_all['applicant_idx']==idx]
            summary.append({'applicant_idx': idx, 'num_cfs': cnt,
                            'num_valid': sub['conforms'].sum(),
                            'min_warning': sub['warning_count'].min(),
                            'avg_distance': sub['distance'].mean(),
                            'avg_cost': sub['cost'].mean()})
        else:
            summary.append({'applicant_idx': idx, 'num_cfs': 0, 'num_valid': 0,
                            'min_warning': 0, 'avg_distance': np.nan, 'avg_cost': np.nan})
    pd.DataFrame(summary).to_csv(f"results/{ds_name}_{args.method}_summary.csv", index=False)
    print("All done.")

if __name__ == "__main__":
    main()