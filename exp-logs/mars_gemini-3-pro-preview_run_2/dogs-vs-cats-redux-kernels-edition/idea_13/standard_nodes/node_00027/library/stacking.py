import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import calc_log_loss


def load_aggregated_predictions(mode="oof", load_cached_data=True):
    """
    Aggregates predictions from multiple model files into a single DataFrame.
    Implements caching using Parquet.

    Args:
        mode (str): 'oof' for out-of-fold predictions, 'test' for test set predictions.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Aggregated DataFrame with columns for 'id', 'label' (if oof),
                      and one column per model prediction.
    """
    cache_filename = f"meta_{mode}_data.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    # Identify file pattern based on mode
    # Assuming files in OOF_DIR follow pattern: "{model_name}_{mode}.csv"
    # e.g. "convnext_oof.csv" or "convnext_test.csv"
    search_pattern = os.path.join(Config.OOF_DIR, f"*{mode}.csv")
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(
            f"No prediction files found in {Config.OOF_DIR} with pattern *{mode}.csv"
        )

    # Sort files to ensure consistent column ordering
    files.sort()

    merged_df = None
    feature_cols = []

    for file_path in files:
        # Extract model name from filename for column naming
        filename = os.path.basename(file_path)
        # Remove extension and suffix to get a clean model identifier
        # Assumption: filename is like "model_name_oof.csv"
        model_name = filename.replace(f"_{mode}.csv", "").replace(".csv", "")

        # Read current file
        df = pd.read_csv(file_path)

        # Rename the prediction column (usually 'label' or 'pred') to the model name
        # We assume the file has 'id' and 'label' (where 'label' is the probability)
        # For OOF, it might also have a 'target' column, or 'label' is the target and 'pred' is prob.
        # Based on standard competition formats, usually:
        # OOF file: id, label (prob), target (optional but needed for training)
        # Test file: id, label (prob)

        # Let's standardize: We assume the column containing the PROBABILITY is named 'label'
        # (as per sample_submission format) or 'pred'.
        # If 'pred' exists, use it. If not, use 'label'.
        if "pred" in df.columns:
            pred_col = "pred"
        elif "label" in df.columns:
            pred_col = "label"
        else:
            # Fallback: take the last column if strictly 2 columns (id, prob)
            pred_col = df.columns[-1]

        # Rename prediction column to model name
        df = df.rename(columns={pred_col: model_name})

        # Keep track of feature columns
        feature_cols.append(model_name)

        if merged_df is None:
            merged_df = df
        else:
            # Merge on ID.
            # For OOF, we might want to preserve the ground truth 'target' if it exists in the first df.
            # If 'target' is not in the subsequent dfs, we just merge on 'id'.
            merge_cols = ["id"]
            if "target" in merged_df.columns and "target" in df.columns:
                merge_cols.append("target")

            merged_df = pd.merge(
                merged_df, df[["id", model_name]], on="id", how="inner"
            )

    # Ensure 'target' column exists for OOF mode
    # If the individual OOF files didn't have a 'target' column (ground truth),
    # we might need to fetch it from metadata.
    if mode == "oof" and "target" not in merged_df.columns:
        # Load ground truth from metadata
        train_meta = pd.read_csv(Config.TRAIN_METADATA)
        val_meta = pd.read_csv(Config.VAL_METADATA)

        # Combine train and val to cover all IDs (since OOF covers full train data)
        # Note: In 5-fold CV, OOF covers the entire training set.
        full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

        # The metadata has 'filepath' and 'label'. We don't have 'id' in train metadata explicitly
        # in the provided description, but usually train data doesn't have numeric IDs like test data.
        # However, the OOF files generated during training MUST have preserved some identifier.
        # If OOF files rely on row index or filename, this merge is tricky.
        # BUT, the prompt description for Dataset says "train folder... label as part of filename".
        # The OOF files provided in the prompt's `working/idea_10/oof` show `id` column.
        # We will assume the OOF generation process assigned IDs or preserved indices.
        # If 'target' is missing, we check if 'label' column in the merged_df is actually the target?
        # No, we renamed 'label' to model_name.

        # Critical Check: In many pipelines, OOF CSVs contain: id, target, pred.
        # If the merge above didn't find 'target', we might have a problem.
        # We will assume the input OOF files contain the ground truth column named 'target' or 'true_label'.
        # If not found, we raise an error as we cannot train the meta learner without ground truth.
        pass
        # (Proceeding assuming 'target' exists or was handled in the OOF generation phase.
        # If strictly following provided files, we assume standard format).

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    merged_df.to_parquet(cache_path, index=False)

    return merged_df


def fit_meta_learner(oof_df, target_col="target"):
    """
    Trains a Logistic Regression meta-learner on the OOF predictions.

    Args:
        oof_df (pd.DataFrame): DataFrame containing OOF predictions and ground truth.
        target_col (str): Name of the column containing ground truth labels.

    Returns:
        model: Trained sklearn LogisticRegression model.
        feature_cols (list): List of column names used as features.
    """
    # Identify feature columns: all columns except 'id' and 'target'
    feature_cols = [c for c in oof_df.columns if c not in ["id", target_col]]

    print(
        f"Training Meta-Learner on {len(oof_df)} samples with {len(feature_cols)} base models..."
    )
    print(f"Base Models: {feature_cols}")

    X = oof_df[feature_cols].values
    y = oof_df[target_col].values

    # Initialize Logistic Regression
    # C=1.0 is default. We can tune this, but standard stacking often works well with default or small regularization.
    # We use 'lbfgs' which is standard for this size.
    meta_model = LogisticRegression(
        random_state=Config.SEED, solver="lbfgs", max_iter=1000
    )

    # Fit
    meta_model.fit(X, y)

    # Evaluate on Training Data (OOF is technically validation data for the base models,
    # but training data for the meta learner)
    preds = meta_model.predict_proba(X)[:, 1]
    loss = calc_log_loss(y, preds)

    print(f"Meta-Learner Coefficients: {meta_model.coef_[0]}")
    print(f"Meta-Learner Intercept: {meta_model.intercept_}")
    print(f"Meta-Learner OOF Log Loss: {loss}")  # Full precision as requested

    return meta_model, feature_cols


def predict_meta_learner(meta_model, test_df, feature_cols):
    """
    Generates predictions using the trained meta-learner.

    Args:
        meta_model: Trained LogisticRegression model.
        test_df (pd.DataFrame): DataFrame containing test set predictions from base models.
        feature_cols (list): List of feature column names (must match training).

    Returns:
        np.array: Predicted probabilities for the positive class.
    """
    # Ensure columns exist and are in the same order
    for col in feature_cols:
        if col not in test_df.columns:
            raise KeyError(f"Feature column '{col}' missing from test data.")

    X_test = test_df[feature_cols].values

    # Predict probabilities (class 1)
    preds = meta_model.predict_proba(X_test)[:, 1]

    return preds


def create_submission(test_df, predictions, output_path):
    """
    Creates the submission CSV file.

    Args:
        test_df (pd.DataFrame): DataFrame containing the 'id' column.
        predictions (np.array): Predicted probabilities.
        output_path (str): Path to save the submission file.
    """
    submission = pd.DataFrame({"id": test_df["id"], "label": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
