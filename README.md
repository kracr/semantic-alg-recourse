# semantic-alg-recourse
Semantically validate the generated counterfactual explanations for algorithmic recourse.

This repository provides a unified pipeline for generating counterfactual explanations with semantic validation. It supports multiple datasets and counterfactual generation methods in a single configurable codebase.

---

## Features

- **Single pipeline** for both datasets: German Credit and COMPAS
- **Multiple methods**: DiCE and ActionRules
- **Config-driven**: All dataset-specific parameters in `config.yaml`
- **Semantic validation**: Uses SHACL constraints and RDFS inference
- **Extensible**: Add new datasets or methods with minimal code changes

---

## Project Structure

```
.
├── code/
│   ├── run_recourse.py          # Unified pipeline
│   └── analyze_results          # Analysis script
├── dataset/
│   ├── german-credit/
│   │   ├── config.yaml
│   │   ├── ontology/             # OWL/RDFS ontology
│   │   ├── shacl/                # SHACL constraints
│   │   └── mapping/              # RML mapping for Morph-KGC
│   └── COMPAS/
│       ├── config.yaml
│       ├── ontology/
│       ├── shacl/
│       └── mapping/
├── models/                       # Saved models (auto-generated)
├── results/                      # Output files (auto-generated)
├── requirements.txt              # Dependencies
└── README.md
```

---

## Dependencies

### Required Packages

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy`, `scipy` | Data processing |
| `scikit-learn`, `joblib` | Machine learning |
| `rdflib`, `pyshacl` | RDF and SHACL validation |
| `morph-kgc` | RDF mapping (RML) |
| `dice-ml` | DiCE counterfactual generation |
| `action-rules` | ActionRules counterfactual generation |
| `matplotlib`, `seaborn` | Plotting and analysis |
| `tqdm` | Progress bars |
| `pyyaml` | Configuration parsing |

### Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn rdflib pyshacl morph-kgc tqdm matplotlib seaborn scipy pyyaml joblib dice-ml action-rules
```

---

## How to Compile / Run

There is no compilation step, since this is a pure Python pipeline. After installing dependencies, run it directly from the repo root.

### Basic Usage

```bash
python code/run_recourse.py --dataset <dataset> --method <method>
```

### German Credit

```bash
# DiCE
python code/run_recourse.py --dataset german-credit --method dice

# ActionRules
python code/run_recourse.py --dataset german-credit --method actionrules
```

### COMPAS

```bash
# DiCE (limited to 100 applicants)
python code/run_recourse.py --dataset compas --method dice --num_cfs 100

# ActionRules (limited to 100 applicants)
python code/run_recourse.py --dataset compas --method actionrules --num_cfs 100
```

### Command-Line Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | `german-credit`, `compas` | Required | Dataset to evaluate |
| `--method` | `dice`, `actionrules` | `actionrules` | CF generation method |
| `--num_cfs` | Integer | `100` | Number of rejected applicants to evaluate |
| `--retrain` | Flag | `False` | Force retraining of the model |

---

## Configuration

Each dataset has a `config.yaml` file in its folder. It specifies:

- **Features**: continuous and categorical feature lists
- **Cost weights**: per-feature cost for distance calculation
- **Immutable attributes**: features that cannot change
- **Artifact paths**: paths to ontology, SHACL, and mapping files
- **ActionRules hyperparameters**: support, confidence, etc.
- **Preprocessing**: method (`dummy` or `standard`) and `stratify` option

Example: `dataset/german-credit/config.yaml`

```yaml
dataset_name: "german-credit"
target: "class"
target_undesired: "bad"
target_desired: "good"

features:
  continuous: ["duration", "credit_amount", ...]
  categorical: ["checking_status", "credit_history", ...]

immutable_attributes: ["personal_status", "foreign_worker", "num_dependents"]

artifact_paths:
  ontology: "ontology/credit_ontology.ttl"
  shacl: "shacl/shacl_rules.ttl"
  rml: "mapping/cf_mapping.yml"

preprocessing:
  method: "dummy"
  stratify: false
```

---

## Output

After running, the following files are generated in `results/`:

| File | Description |
|------|-------------|
| `{dataset}_{method}_validation.csv` | Per-CF validation table |
| `{dataset}_{method}_summary.csv` | Per-applicant summary |
| `{dataset}_{method}_counterfactuals.ttl` | RDF graph with explanation triples |

---

## Analysis

After running the pipeline, analyze the results:

```bash
python code/analyze_results --validation_file results/german-credit_dice_validation.csv
```

This generates:
- Violation counts per SHACL rule
- Distance and cost comparison (valid vs invalid)
- Plots and summary statistics

---

## Extending the Pipeline

### Adding a New Dataset

1. Create a folder under `dataset/`: `dataset/newdataset/`
2. Add `config.yaml` with features, paths, and parameters
3. Place ontology, SHACL, and mapping files in subfolders
4. Run: `python code/run_recourse.py --dataset newdataset --method dice`

### Adding a New Method

1. Add a new function `gen_cfs_<method>()` in `run_recourse.py`
2. Register it in the method selection block
3. Add the method name to the argument choices

---

## License

This project is licensed under the Apache License 2.0 – see the LICENSE file for details.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainers.
