from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathsnap.evaluation.model_evaluation import evaluate_model

def get_model():
    """
    Returns Logistic Regression Pipeline with StandardScaler.
    Algorithm Family: Linear and probabilistic.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

def train_and_evaluate(train_df, test_df, save_path=None):
    model = get_model()
    result = evaluate_model(
        model_name="Logistic Regression",
        algorithm_family="Linear and Probabilistic",
        key_config="C=1.0, StandardScaler, balanced class_weight",
        model=model,
        train_df=train_df,
        test_df=test_df
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")
    return result, model
