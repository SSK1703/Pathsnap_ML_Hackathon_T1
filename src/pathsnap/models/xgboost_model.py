from pathlib import Path
import joblib
import xgboost as xgb

from pathsnap.evaluation.model_evaluation import evaluate_model

def get_model():
    """
    Returns XGBoost Classifier.
    Algorithm Family: Boosting.
    """
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=9.0, # accounts for ~1:9 positive:negative imbalance
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

def train_and_evaluate(train_df, test_df, save_path=None):
    model = get_model()
    result = evaluate_model(
        model_name="XGBoost",
        algorithm_family="Boosting",
        key_config="n_est=100, depth=6, lr=0.1, scale_pos_weight=9.0",
        model=model,
        train_df=train_df,
        test_df=test_df
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")
    return result, model
