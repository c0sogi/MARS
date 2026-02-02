import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

# Import library modules
from library import config, data_processing, dataset, models, training, evaluation


def main():
    # Set seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # -------------------------------------------------------------------------
    print("Configuring for fast baseline...")
    # Limit training epochs for speed
    config.TRAIN_PARAMS["epochs"] = 10
    # Limit training samples (Subsampling) to ensure quick execution
    config.TRAIN_PARAMS["debug_sample_size"] = 500000

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    # We generate training data WITH subsampling, and validation data WITHOUT subsampling.

    # A. Generate Training Data (Subsampled)
    print("Generating subsampled training data...")
    # Force regeneration (load_cached_data=False) to apply the debug_sample_size
    # This also fits and saves the scaler.
    data_processing.get_train_data(load_cached_data=False)

    # B. Generate Validation Data (Full)
    # Reset sample size to None to ensure we validate on the full hold-out set
    config.TRAIN_PARAMS["debug_sample_size"] = None

    print("Generating full validation data...")
    # Force regeneration to ensure full set is processed using the scaler from step A
    data_processing.get_val_data(load_cached_data=False)

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("Starting training...")
    # training.train_model() initializes loaders (using the caches we just created)
    # and runs the training loop with the GRVNet model.
    training.train_model()

    # -------------------------------------------------------------------------
    # 4. Final Evaluation on Validation Set
    # -------------------------------------------------------------------------
    print("Performing final validation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model saved during training
    model = models.GRVNet().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model file not found. Training might have failed.")
        return

    # Get Validation Loader (Full set, as cached)
    _, val_loader = dataset.get_train_val_loaders()

    # Optimize Threshold and Get Metric
    # This function runs inference on the validation set and finds the best MCC
    best_threshold, best_mcc = evaluation.optimize_threshold(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {best_mcc}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("Running failure analysis...")
    # Get predictions and targets for the validation set
    probs, targets = evaluation.run_inference(model, val_loader, device)

    # Calculate absolute error (|Target - Prediction|)
    errors = np.abs(targets - probs)

    # Load raw feature arrays to calculate correlations
    X_kin_val, X_vis_val, _, _ = data_processing.get_val_data(load_cached_data=True)

    # Construct Feature Names to match the column order in X_kin and X_vis
    kin_names = []
    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        for feat in config.KINEMATIC_FEATURES:
            kin_names.append(f"{feat}_lag_{lag}")

    vis_names = []
    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        for feat in config.VISUAL_FEATURES:
            vis_names.append(f"{feat}_lag_{lag}")

    all_feature_names = kin_names + vis_names

    # Stack features horizontally
    X_all = np.hstack([X_kin_val, X_vis_val])

    # Calculate correlations between each feature and the error
    print(f"Calculating correlations for {X_all.shape[1]} features...")
    correlations = []
    for i in range(X_all.shape[1]):
        feat_col = X_all[:, i]
        # Handle constant columns (std=0) to avoid NaN
        if np.std(feat_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append(corr)

    # Create DataFrame for analysis
    corr_df = pd.DataFrame({"Feature": all_feature_names, "Correlation": correlations})

    # Sort by absolute correlation to find features most strongly associated with error
    corr_df["AbsCorr"] = corr_df["Correlation"].abs()
    top_corrs = corr_df.sort_values("AbsCorr", ascending=False).head(10)

    print("\nTop 10 Features associated with Error:")
    print(top_corrs[["Feature", "Correlation"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.6634847318478787

    if best_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation MCC ({best_mcc}) > {THRESHOLD_SCORE}. Generating submission..."
        )
        # evaluation.generate_predictions loads the best model, runs inference on the test set,
        # applies the optimized threshold, and saves submission.csv
        evaluation.generate_predictions()
    else:
        print(
            f"\nValidation MCC ({best_mcc}) <= {THRESHOLD_SCORE}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
