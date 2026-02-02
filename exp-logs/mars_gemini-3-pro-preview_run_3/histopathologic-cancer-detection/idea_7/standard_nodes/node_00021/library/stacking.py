import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config


def load_or_create_oof_dataset(
    predictions_dict=None, targets=None, load_cached_data=True
):
    """
    Loads the stacked OOF dataset from cache or creates it from raw predictions.
    Strictly follows the caching logic required.

    Args:
        predictions_dict (dict, optional): Dictionary {model_name: np.array(probs)}.
        targets (np.array, optional): Ground truth labels.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing model predictions and 'target' column.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = Config.STACKED_OOF_DATA

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            # Fall through to scratch creation if load fails
            pass

    # 2. Create from scratch
    if predictions_dict is None or targets is None:
        raise ValueError(
            "predictions_dict and targets are required to create OOF dataset when cache is not used."
        )

    # Ensure deterministic column order by sorting model names
    model_names = sorted(predictions_dict.keys())

    data = {}
    for name in model_names:
        data[name] = predictions_dict[name]

    df = pd.DataFrame(data)
    df["target"] = targets

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


def train_meta_learner(df, save_path=Config.META_LEARNER_PATH):
    """
    Trains a Logistic Regression meta-learner on the OOF predictions.

    Args:
        df (pd.DataFrame): DataFrame containing feature columns and 'target'.
        save_path (str): Path to save the trained model.

    Returns:
        model: Trained sklearn model.
        float: ROC AUC score.
    """
    # Identify feature columns (all except 'target')
    # Sort to ensure consistency with how data was created/will be predicted
    feature_cols = sorted([c for c in df.columns if c != "target"])

    X = df[feature_cols].values
    y = df["target"].values

    # Initialize and train Logistic Regression
    # Using lbfgs solver which is standard for this size
    model = LogisticRegression(random_state=Config.SEED, solver="lbfgs")
    model.fit(X, y)

    # Evaluate on the training set (OOF)
    # This represents the expected performance of the stack
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    print(f"Meta-Learner Training Complete. OOF AUC: {auc}")

    # Save model
    joblib.dump(model, save_path)

    return model, auc


def predict_stacked(predictions_dict, model_path=Config.META_LEARNER_PATH):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        predictions_dict (dict): Dictionary {model_name: np.array(probs)} for test set.
        model_path (str): Path to the saved meta-learner.

    Returns:
        np.array: Final calibrated probabilities.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Meta-learner model not found at {model_path}")

    model = joblib.load(model_path)

    # Prepare features ensuring same order as training
    model_names = sorted(predictions_dict.keys())

    # Construct feature matrix
    # Stack arrays as columns
    X = np.column_stack([predictions_dict[name] for name in model_names])

    # Predict
    final_probs = model.predict_proba(X)[:, 1]

    return final_probs
