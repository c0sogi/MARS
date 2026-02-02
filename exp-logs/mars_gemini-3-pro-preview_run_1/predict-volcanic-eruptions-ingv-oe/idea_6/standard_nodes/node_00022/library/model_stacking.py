import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import seed_everything, get_score


class MetaLearner:
    """
    Implements the Stacking Meta-Learner using Ridge Regression.
    Combines predictions from Tabular and Vision branches.
    """

    def __init__(self):
        self.config = Config()
        self.model = Ridge(
            alpha=self.config.META_PARAMS["alpha"],
            random_state=self.config.META_PARAMS["random_state"],
        )

    def train(self, X, y):
        """
        Trains the Ridge Regression model.

        Args:
            X (pd.DataFrame or np.ndarray): Input features (predictions from base models).
            y (pd.Series or np.ndarray): Ground truth targets.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Predictions.
        """
        preds = self.model.predict(X)
        # Ensure non-negative predictions for time_to_eruption
        return np.maximum(0, preds)


def load_and_merge_predictions(
    tabular_path, vision_path, metadata_paths=None, is_train=True
):
    """
    Loads prediction CSVs and optionally merges with metadata for ground truth.

    Args:
        tabular_path (str): Path to tabular predictions CSV.
        vision_path (str): Path to vision predictions CSV.
        metadata_paths (list): List of paths to metadata CSVs (for ground truth).
        is_train (bool): Whether loading training (OOF) or test data.

    Returns:
        pd.DataFrame: Merged DataFrame with 'pred_tabular', 'pred_vision', and optionally 'target'.
    """
    # Load predictions
    if not os.path.exists(tabular_path):
        raise FileNotFoundError(f"Tabular predictions not found at {tabular_path}")
    if not os.path.exists(vision_path):
        raise FileNotFoundError(f"Vision predictions not found at {vision_path}")

    df_tab = pd.read_csv(tabular_path)
    df_vis = pd.read_csv(vision_path)

    # Rename columns to avoid collision and clarify source
    df_tab = df_tab.rename(columns={"time_to_eruption": "pred_tabular"})
    df_vis = df_vis.rename(columns={"time_to_eruption": "pred_vision"})

    # Merge on segment_id
    df_merged = pd.merge(df_tab, df_vis, on="segment_id", how="inner")

    if is_train and metadata_paths:
        # Load and concatenate metadata to get ground truth
        meta_dfs = []
        for path in metadata_paths:
            if os.path.exists(path):
                meta_dfs.append(pd.read_csv(path))
            else:
                raise FileNotFoundError(f"Metadata file not found at {path}")

        df_meta = pd.concat(meta_dfs, ignore_index=True)

        # Keep only relevant columns from metadata
        df_meta = df_meta[["segment_id", "time_to_eruption"]]
        df_meta = df_meta.rename(columns={"time_to_eruption": "target"})

        # Merge with predictions
        df_merged = pd.merge(df_merged, df_meta, on="segment_id", how="inner")

    return df_merged


def run_stacking(debug=False):
    """
    Orchestrates the Meta-Learner training and submission generation.

    Args:
        debug (bool): If True, runs on a subset (though usually fast enough to run full).
    """
    config = Config()
    seed_everything(config.SEED)

    print("Initializing Meta-Learner (Stacking)...")

    working_dir = config.WORKING_DIR
    submission_dir = config.SUBMISSION_DIR

    # ==========================================
    # 1. Define Paths
    # ==========================================
    # Input OOF files
    tab_oof_path = os.path.join(working_dir, "tabular_oof.csv")
    vis_oof_path = os.path.join(working_dir, "vision_oof.csv")

    # Input Test Prediction files
    tab_test_path = os.path.join(working_dir, "tabular_test_preds.csv")
    vis_test_path = os.path.join(working_dir, "vision_test_preds.csv")

    # Metadata files
    train_meta_path = config.TRAIN_METADATA_PATH
    val_meta_path = config.VAL_METADATA_PATH
    test_meta_path = config.TEST_METADATA_PATH

    # ==========================================
    # 2. Process Training Data (OOF)
    # ==========================================
    print("Loading and merging OOF predictions...")
    df_train = load_and_merge_predictions(
        tab_oof_path,
        vis_oof_path,
        metadata_paths=[train_meta_path, val_meta_path],
        is_train=True,
    )

    X_train = df_train[["pred_tabular", "pred_vision"]]
    y_train = df_train["target"]

    if debug:
        print("DEBUG: Subsampling training data for meta-learner.")
        X_train = X_train.iloc[:100]
        y_train = y_train.iloc[:100]

    # ==========================================
    # 3. Train Meta-Learner
    # ==========================================
    print("Training Ridge Regression Meta-Learner...")
    meta_model = MetaLearner()
    meta_model.train(X_train, y_train)

    # Evaluate on Training Data (OOF) to check fit
    # Note: This is technically evaluating on OOF data, which is a valid proxy for generalization
    # if the OOFs were generated correctly.
    preds_train = meta_model.predict(X_train)
    score = get_score(y_train, preds_train)
    print(f"Meta-Learner OOF MAE: {score}")

    # Print learned coefficients
    coefs = meta_model.model.coef_
    intercept = meta_model.model.intercept_
    print(f"Learned Coefficients: Tabular={coefs[0]}, Vision={coefs[1]}")
    print(f"Intercept: {intercept}")

    # ==========================================
    # 4. Process Test Data and Predict
    # ==========================================
    print("Loading and merging Test predictions...")
    # We load test metadata just to ensure we have the list of segment_ids required for submission
    # though the prediction files should already have them.
    df_test_preds = load_and_merge_predictions(
        tab_test_path, vis_test_path, is_train=False
    )

    # Load test metadata to ensure correct order/inclusion of all segments
    df_test_meta = pd.read_csv(test_meta_path)

    # Merge with metadata to ensure we have all segments and correct order
    df_final = pd.merge(
        df_test_meta[["segment_id"]], df_test_preds, on="segment_id", how="left"
    )

    # Fill missing predictions if any (should not happen if pipeline is correct) with 0 or mean
    if df_final.isnull().any().any():
        print("Warning: Missing predictions for some test segments. Filling with 0.")
        df_final = df_final.fillna(0)

    X_test = df_final[["pred_tabular", "pred_vision"]]

    print("Generating final predictions...")
    final_predictions = meta_model.predict(X_test)

    # ==========================================
    # 5. Create Submission File
    # ==========================================
    submission_df = pd.DataFrame(
        {"segment_id": df_final["segment_id"], "time_to_eruption": final_predictions}
    )

    # Ensure output directory exists
    os.makedirs(submission_dir, exist_ok=True)
    save_path = config.SUBMISSION_PATH

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())
