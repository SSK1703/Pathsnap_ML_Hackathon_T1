# PathSnap - Machine Learning Hackathon

**Project:** PathSnap: Distributed Map-Matching and Route Reconstruction

**Course:** UE24CS352A - Machine Learning

**Team Size:** 4 Members

---

## 1. Project Overview

PathSnap is a machine learning system designed to reconstruct realistic
vehicle trajectories by matching noisy GPS observations to a road network.

GPS observations can contain errors due to:
- Urban building interference
- Signal multipath effects
- GPS noise
- Signal loss
- Irregular sampling intervals

The objective is to identify the most likely road segments traversed by
a vehicle using GPS observations and road-network information.

The system will eventually include:

1. Dataset exploration and preprocessing
2. Feature engineering and candidate generation
3. Multiple ML model experiments
4. Model comparison and selection
5. Bayesian hyperparameter optimization
6. HMM-based route reconstruction
7. Viterbi decoding
8. FastAPI backend
9. Evaluator-facing UI
10. Geographical visualization using Folium

---

## 2. Dataset Structure

The supplied dataset contains four primary file types.

| File | Description |
|------|-------------|
| .track | GPS observations |
| .nodes | Road-network geographical nodes |
| .arcs | Connections between road-network nodes |
| .route | Reference sequence of road-segment IDs |

### File Formats

.track:
- Longitude
- Latitude
- Timestamp in seconds

.nodes:
- Longitude
- Latitude
- Row index represents node ID

.arcs:
- From node ID
- To node ID
- Row index represents arc ID

.route:
- Chronological sequence of reference arc IDs

All files are whitespace-separated and headerless according
to the provided dataset guidelines.

---

## 3. Dataset Partitions

| Partition | Number of Trips | Purpose |
|-----------|----------------|---------|
| Training | 80 | Model training, development and cross-validation |
| Student Testing | 10 | Development-stage evaluation |
| Evaluator Testing | 10 | Final evaluation only |

Evaluator-testing data must not be used for training, model selection,
feature selection or hyperparameter optimization.

All observations from a trip must remain within the same partition.

---

## 4. Project Structure

```text
config/           Configuration files
data/raw/         Original dataset
data/processed/   Processed dataset
notebooks/        Jupyter notebooks
src/pathsnap/     Reusable project modules
scripts/          Executable scripts
tests/            Unit tests
reports/          Report assets
artifacts/        Final model artifacts
```

## 5. Model Experiments

Install the project dependencies and make the package importable:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Place the training and student-test trip folders under
`src/pathsnap/data/raw/train/` and `src/pathsnap/data/raw/student_test/`.
Then build candidates and run the five model comparison:

```bash
python scripts/build_dataset.py
python scripts/train_models.py
```

The training pipeline evaluates using five-fold `GroupKFold` by trip. It
reports student-test metrics only when the processed student-test file has
`is_match` labels; otherwise it reports inference time without labeling the
test set. Bayesian Random Forest tuning can be run with
`python scripts/optuna_search.py`.

The notebook `notebooks/02_model_experimentation.ipynb` contains the
interactive comparison. Processed data and trained model artifacts are
generated locally and should not be committed.
