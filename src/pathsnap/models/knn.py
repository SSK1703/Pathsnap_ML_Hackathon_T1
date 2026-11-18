from pathlib import Path
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathsnap.evaluation.model_evaluation import evaluate_model

def get_model():
    """
    Returns Distance-Weighted KNN Pipeline with StandardScaler.
    Algorithm Family: Instance-based.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            algorithm="kd_tree",
            n_jobs=-1
        ))
    ])

def train_and_evaluate(train_df, test_df, save_path=None):
    model = get_model()
    result = evaluate_model(
        model_name="KNN (Distance-Weighted)",
        algorithm_family="Instance-based",
        key_config="k=7, distance-weighted, StandardScaler, kd_tree",
        model=model,
        train_df=train_df,
        test_df=test_df
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")
    return result, model
