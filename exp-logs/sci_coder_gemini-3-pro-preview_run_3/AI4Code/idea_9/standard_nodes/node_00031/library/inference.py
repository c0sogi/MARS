import os
import pandas as pd
import lightgbm as lgb
import torch
import numpy as np
import random
from library.config import Config
from library.data_loader import get_data_splits
from library.feature_engineering import generate_features_pipeline
from library.regressor import reconstruct_orders


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_submission_file(load_cached_data: bool = True):
    """
    Orchestrates the test set prediction workflow.

    1. Loads test metadata.
    2. Generates (or loads cached) multi-scale features for test notebooks.
    3. Loads the trained LightGBM regressor.
    4. Predicts ranking scores for markdown cells.
    5. Reconstructs the full cell order (interleaving code and markdown).
    6. Saves the final submission file.

    Args:
        load_cached_data (bool): If True, attempts to load intermediate feature files
                                 from disk to save compute time.
    """
    # 1. Setup and Initialization
    set_seed(Config.SEED)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Initializing Inference Pipeline...")

    # 2. Load Metadata
    # We only need the test split for inference
    _, _, df_test_meta = get_data_splits()
    print(f"Loaded metadata for {len(df_test_meta)} test notebooks.")

    # 3. Feature Engineering
    # This pipeline handles loading the fine-tuned SentenceTransformer backbone,
    # computing embeddings, applying multi-scale smoothing, and caching the result.
    print("Generating/Loading Test Features...")
    df_test_feats = generate_features_pipeline(
        df_metadata=df_test_meta,
        mode="test",
        model=None,  # Pipeline will load model from Config.MODEL_OUTPUT_PATH
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # 4. Model Loading and Prediction
    if not os.path.exists(Config.LGBM_MODEL_PATH):
        raise FileNotFoundError(
            f"LightGBM model not found at {Config.LGBM_MODEL_PATH}. "
            "Please ensure the regression model is trained before running inference."
        )

    print(f"Loading LightGBM model from {Config.LGBM_MODEL_PATH}")
    bst = lgb.Booster(model_file=Config.LGBM_MODEL_PATH)

    if not df_test_feats.empty:
        # Filter columns to only include feature vectors (exclude metadata columns)
        # Metadata columns are 'id', 'cell_id', and 'target' (though target is not in test)
        feature_cols = [
            c for c in df_test_feats.columns if c not in ["id", "cell_id", "target"]
        ]

        print(f"Predicting ranks using {len(feature_cols)} features...")
        X_test = df_test_feats[feature_cols].values

        # LightGBM prediction
        preds = bst.predict(X_test)
        df_test_feats["pred"] = preds
    else:
        print(
            "Warning: No test features generated. Submission will contain default orders."
        )
        # Create an empty DataFrame with required columns to allow graceful failure in reconstruction
        df_test_feats = pd.DataFrame(columns=["id", "cell_id", "pred"])

    # 5. Order Reconstruction
    # Converts predicted scores and code cell skeletons into the final space-delimited string
    print("Reconstructing cell orders...")
    df_submission = reconstruct_orders(
        df_preds=df_test_feats, df_metadata=df_test_meta, mode="test"
    )

    # 6. Save Submission
    print(f"Saving submission file to {Config.SUBMISSION_PATH}")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Inference completed successfully.")
