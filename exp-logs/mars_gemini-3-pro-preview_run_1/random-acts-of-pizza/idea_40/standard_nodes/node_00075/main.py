import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import local library modules
from library import config, utils, features, rf_model, training, dataset


def run_failure_analysis(y_true, y_pred, feature_data, feature_names=None):
    """
    Analyzes the correlation between prediction error and input features.
    Focuses on dense metadata features for interpretability.
    """
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - y_pred)

    # We will analyze correlations with the metadata features provided in the MLP stream
    # as they are the most interpretable dense features (account age, karma, etc.)
    # feature_data is expected to be (N, D)

    correlations = []
    num_features = feature_data.shape[1]

    for i in range(num_features):
        feat_vals = feature_data[:, i]
        # Handle constant features to avoid warning
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        feat_name = f"Feature_{idx}" if feature_names is None else feature_names[idx]
        print(f"{feat_name}: {corr:.4f}")


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(config.RANDOM_STATE)
    print("Starting Hybrid Ensemble Pipeline...")

    # 2. Data Loading & Feature Engineering
    # We use the FeatureEngineer directly to get access to validation data for ensembling
    print("Loading and processing data...")
    fe = features.FeatureEngineer()
    rf_out, mlp_out = fe.process_data(load_cached_data=True)

    # Extract RF Data
    X_train_rf = rf_out["X_train"]
    y_train = rf_out["y_train"]
    X_val_rf = rf_out["X_val"]
    y_val = rf_out["y_val"]
    X_test_rf = rf_out["X_test"]
    request_ids_test = rf_out["request_ids_test"]

    # 3. Stream A: Random Forest Pipeline
    print("\n--- Stream A: Random Forest ---")
    # Train RF
    rf_clf, rf_val_auc = rf_model.train_rf(X_train_rf, y_train, X_val_rf, y_val)

    # Inference (RF)
    val_probs_rf = rf_clf.predict_proba(X_val_rf)[:, 1]
    test_probs_rf = rf_clf.predict_proba(X_test_rf)[:, 1]

    # 4. Stream B: MLP Pipeline
    print("\n--- Stream B: Skip-Gated MLP ---")
    # Train MLP
    # We use the provided training function which handles the training loop and best model saving
    mlp_results = training.train_mlp_model(load_cached_data=True)
    best_mlp_model = mlp_results["model"]
    test_probs_mlp = mlp_results["test_preds"]

    # Inference (MLP Validation)
    # We need to manually run inference on the validation set to get probabilities for the ensemble
    print("Generating MLP validation predictions for ensemble...")
    _, val_dataset, _ = dataset.get_mlp_datasets(load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.MLP_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    device = next(best_mlp_model.parameters()).device
    best_mlp_model.eval()
    val_probs_mlp = []

    with torch.no_grad():
        for batch in val_loader:
            title_emb = batch["title_emb"].to(device)
            body_emb = batch["body_emb"].to(device)
            hist_seq = batch["hist_seq"].to(device)
            hist_mask = batch["hist_mask"].to(device)
            meta = batch["meta"].to(device)
            cons = batch["cons"].to(device)

            logits = best_mlp_model(
                title_emb, body_emb, hist_seq, hist_mask, meta, cons
            )
            probs = torch.sigmoid(logits)
            val_probs_mlp.extend(probs.cpu().numpy())

    val_probs_mlp = np.array(val_probs_mlp)

    # 5. Ensemble & Validation
    print("\n--- Ensemble Evaluation ---")
    # Simple weighted average
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    val_probs_ensemble = (w_rf * val_probs_rf) + (w_mlp * val_probs_mlp)

    final_val_auc = roc_auc_score(y_val, val_probs_ensemble)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    # We use the metadata features from the MLP stream for analysis as they are clean numerical features
    run_failure_analysis(y_val, val_probs_ensemble, mlp_out["val_meta"])

    # 7. Submission
    target_threshold = 0.7056961514236341
    if final_val_auc > target_threshold:
        print(
            f"\nValidation score ({final_val_auc}) exceeds threshold ({target_threshold}). Generating submission..."
        )

        test_probs_ensemble = (w_rf * test_probs_rf) + (w_mlp * test_probs_mlp)

        utils.save_submission(request_ids_test, test_probs_ensemble)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score ({final_val_auc}) did not exceed threshold ({target_threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
