from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

from pathsnap.evaluation.model_evaluation import evaluate_model

def get_model():
    """
    Returns Random Forest Classifier.
    Algorithm Family: Ensemble.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

def train_and_evaluate(train_df, test_df, save_path=None):
    model = get_model()
    result = evaluate_model(
        model_name="Random Forest",
        algorithm_family="Ensemble",
        key_config="n_estimators=100, max_depth=12, balanced, n_jobs=-1",
        model=model,
        train_df=train_df,
        test_df=test_df
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")
    return result, model
