import os
import numpy as np
import pandas as pd
from library.config import Config
from library.feature_engineering import get_features
from library.models_rf import RFModelWrapper
from library.models_mlp import MLPModelWrapper

# Set deterministic seeds
np.random.seed(Config.RANDOM_STATE)


def _subset_data(data_dict, labels_dict, n_samples):
    """
    Helper to slice data dictionaries for debugging/testing purposes.
    Handles both numpy arrays and sparse matrices.
    """
    if n_samples is None:
        return data_dict, labels_dict

    # Slice Feature Data
    for k, v in data_dict.items():
        if v is not None:
            # Only slice train and validation sets
            if "train" in k or "val" in k:
                current_size = v.shape[0]
                if current_size > n_samples:
                    data_dict[k] = v[:n_samples]

    # Slice Labels
    if labels_dict is not None:
        for k, v in labels_dict.items():
            if v is not None:
                current_size = v.shape[0]
                if current_size > n_samples:
                    labels_dict[k] = v[:n_samples]

    return data_dict, labels_dict


def run_training(load_cached_data=True, epochs=None, sample_size=None):
    """
    Orchestrates the training of the Hybrid Ensemble (RF + MLP).

    Args:
        load_cached_data (bool): Whether to load features from cache.
        epochs (int, optional): Override default epochs for MLP.
        sample_size (int, optional): Number of samples to use for training/val (debugging).

    Returns:
        tuple: (rf_val_auc, mlp_val_auc)
    """
    # 1. Dynamic Configuration Update
    if epochs is not None:
        Config.MLP_PARAMS["epochs"] = epochs

    print(
        f"Starting training pipeline (Epochs={Config.MLP_PARAMS['epochs']}, Sample Size={sample_size})..."
    )

    # 2. Load Features
    # get_features handles caching internally based on the flag
    rf_data, mlp_data, labels = get_features(load_cached_data=load_cached_data)

    # 3. Apply Subsetting (if requested for debugging)
    if sample_size is not None:
        print(f"Subsetting data to {sample_size} samples...")
        rf_data, labels = _subset_data(rf_data, labels, sample_size)
        mlp_data, _ = _subset_data(mlp_data, None, sample_size)  # Labels already sliced

    # 4. Stream A: Random Forest
    print("\n=== Stream A: Random Forest ===")
    rf_model = RFModelWrapper()
    rf_val_auc = rf_model.train(rf_data, labels)
    rf_test_preds = rf_model.predict(rf_data)

    # 5. Stream B: MLP (Dual-Query Masked-Attention)
    print("\n=== Stream B: MLP ===")
    mlp_model = MLPModelWrapper()
    mlp_val_auc = mlp_model.train(mlp_data, labels)
    mlp_test_preds = mlp_model.predict(mlp_data)

    # 6. Ensemble
    print("\n=== Ensembling ===")
    weights = Config.ENSEMBLE_WEIGHTS
    w_rf = weights["rf"]
    w_mlp = weights["mlp"]

    # Normalize weights
    total_weight = w_rf + w_mlp
    w_rf /= total_weight
    w_mlp /= total_weight

    print(f"Weights -> RF: {w_rf:.2f}, MLP: {w_mlp:.2f}")
    final_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

    # 7. Generate Submission
    print("\n=== Generating Submission ===")
    # Load test IDs from metadata
    if os.path.exists(Config.TEST_PATH):
        test_df = pd.read_csv(Config.TEST_PATH)

        # Ensure lengths match
        if len(test_df) != len(final_preds):
            print(
                f"Warning: Length mismatch. Test IDs: {len(test_df)}, Preds: {len(final_preds)}"
            )
            # In debug mode with subsetting, this might happen.
            # We assume standard run matches.
            test_df = test_df.iloc[: len(final_preds)]

        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    else:
        print(f"Error: Test metadata not found at {Config.TEST_PATH}")

    # 8. Final Report
    print("\n=== Final Validation Metrics ===")
    print(f"RF Validation AUC: {rf_val_auc}")
    print(f"MLP Validation AUC: {mlp_val_auc}")

    return rf_val_auc, mlp_val_auc
