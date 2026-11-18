from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

from pathsnap.evaluation.model_evaluation import evaluate_model

def get_model():
    """
    Returns Decision Tree Classifier.
    Algorithm Family: Tree-based.
    """
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )

def train_and_evaluate(train_df, test_df, save_path=None):
    model = get_model()
    result = evaluate_model(
        model_name="Decision Tree",
        algorithm_family="Tree-based",
        key_config="max_depth=10, min_samples_leaf=10, balanced",
        model=model,
        train_df=train_df,
        test_df=test_df
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")
    return result, model
