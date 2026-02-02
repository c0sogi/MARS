import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.data_factory import load_data, prepare_nn_data, get_cv_folds, ForestDataset
from library.xgb_trainer import train_xgb_fold, predict_xgb
from library.nn_trainer import train_nn_fold, predict_nn
from library.ensemble_optimizer import optimize_blending_weights, weighted_average


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing Hybrid Ensemble Workflow...")

    # 2. Data Loading
    # Load data for XGBoost (DataFrames with interactions)
    print("Loading data for XGBoost...")
    X_train_xgb, y_train, X_test_xgb, test_ids = load_data(load_cached_data=True)

    # Load data for Neural Network (Scaled Numpy Arrays)
    print("Loading data for Neural Network...")
    X_train_nn, _, X_test_nn, _ = prepare_nn_data(load_cached_data=True)

    # Get Folds
    folds = get_cv_folds(y_train, n_folds=Config.N_FOLDS, seed=Config.SEED)

    # Initialize storage
    # OOF Predictions: (N_samples, N_classes)
    oof_preds_xgb = np.zeros((len(y_train), Config.NUM_CLASSES))
    oof_preds_nn = np.zeros((len(y_train), Config.NUM_CLASSES))

    # Test Predictions: (N_test_samples, N_classes) - we will sum and divide by N_FOLDS
    test_preds_xgb_sum = np.zeros((len(test_ids), Config.NUM_CLASSES))
    test_preds_nn_sum = np.zeros((len(test_ids), Config.NUM_CLASSES))

    # 3. Training Loop
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{Config.N_FOLDS} ---")

        # --- XGBoost Branch ---
        print("Training XGBoost...")
        # Slice data
        X_tr_xgb = X_train_xgb.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val_xgb = X_train_xgb.iloc[val_idx]
        y_val = y_train[val_idx]

        # Train
        xgb_model = train_xgb_fold(X_tr_xgb, y_tr, X_val_xgb, y_val)

        # Inference (OOF)
        oof_preds_xgb[val_idx] = predict_xgb(xgb_model, X_val_xgb)

        # Inference (Test)
        test_preds_xgb_sum += predict_xgb(xgb_model, X_test_xgb)

        # Cleanup XGB
        del xgb_model, X_tr_xgb, X_val_xgb
        gc.collect()

        # --- Neural Network Branch ---
        print("Training Neural Network...")
        # Slice data
        X_tr_nn = X_train_nn[train_idx]
        X_val_nn = X_train_nn[val_idx]
        # y_tr and y_val are same as above

        # Create Datasets
        train_dataset = ForestDataset(X_tr_nn, y_tr)
        val_dataset = ForestDataset(X_val_nn, y_val)

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.NN_PARAMS["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.NN_PARAMS["batch_size"] * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Train
        nn_model = train_nn_fold(
            train_loader,
            val_loader,
            input_dim=X_tr_nn.shape[1],
            num_classes=Config.NUM_CLASSES,
        )

        # Inference (OOF)
        # Re-use val_loader for consistency
        oof_preds_nn[val_idx] = predict_nn(nn_model, val_loader)

        # Inference (Test)
        test_dataset = ForestDataset(X_test_nn, y=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.NN_PARAMS["batch_size"] * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        test_preds_nn_sum += predict_nn(nn_model, test_loader)

        # Cleanup NN
        del (
            nn_model,
            train_loader,
            val_loader,
            test_loader,
            train_dataset,
            val_dataset,
            X_tr_nn,
            X_val_nn,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # Average Test Predictions
    test_preds_xgb_avg = test_preds_xgb_sum / Config.N_FOLDS
    test_preds_nn_avg = test_preds_nn_sum / Config.N_FOLDS

    # 4. Ensemble Optimization
    print("\n--- Optimizing Ensemble Weights ---")
    oof_dict = {"XGBoost": oof_preds_xgb, "NeuralNet": oof_preds_nn}

    weights_dict = optimize_blending_weights(oof_dict, y_train)

    # Calculate Final Weighted OOF
    final_oof_probs = weighted_average(
        [oof_preds_xgb, oof_preds_nn],
        [weights_dict["XGBoost"], weights_dict["NeuralNet"]],
    )
    final_oof_preds = np.argmax(final_oof_probs, axis=1)

    # 5. Validation & Failure Analysis
    final_acc = accuracy_score(y_train, final_oof_preds)
    print(f"Final Validation Metric: {final_acc}")

    print("\n--- Failure Analysis ---")
    # Identify errors
    errors = (final_oof_preds != y_train).astype(int)
    error_rate = errors.mean()
    print(f"Overall Error Rate: {error_rate:.5f}")

    # Calculate correlation of features with error
    # We use the XGBoost DataFrame as it has column names
    # We'll calculate correlation for a subset of features to save time/memory if needed,
    # but with 2.8M rows, we can do it efficiently.
    print("Calculating feature correlations with error...")

    # Add error column temporarily to compute correlation
    # Use a sample for correlation if dataset is huge to speed up
    sample_size = min(500000, len(X_train_xgb))
    sample_indices = np.random.choice(len(X_train_xgb), sample_size, replace=False)

    df_analysis = X_train_xgb.iloc[sample_indices].copy()
    df_analysis["error_flag"] = errors[sample_indices]

    # Compute correlations with 'error_flag'
    correlations = (
        df_analysis.corrwith(df_analysis["error_flag"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 Features correlated with Error:")
    print(correlations.head(11).iloc[1:])  # Skip self-correlation

    del df_analysis
    gc.collect()

    # 6. Submission
    THRESHOLD = 0.9615416559837934

    if final_acc > THRESHOLD:
        print(
            f"\nValidation Metric ({final_acc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Calculate Weighted Test Predictions
        final_test_probs = weighted_average(
            [test_preds_xgb_avg, test_preds_nn_avg],
            [weights_dict["XGBoost"], weights_dict["NeuralNet"]],
        )
        final_test_preds = np.argmax(final_test_probs, axis=1)

        # Map back to original labels
        # Config.ORIGINAL_LABELS = [1, 2, 3, 4, 6, 7]
        # Map 0->1, 1->2, 2->3, 3->4, 4->6, 5->7
        label_map = {i: label for i, label in enumerate(Config.ORIGINAL_LABELS)}
        final_submission_labels = np.vectorize(label_map.get)(final_test_preds)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_submission_labels}
        )

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation Metric ({final_acc}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
