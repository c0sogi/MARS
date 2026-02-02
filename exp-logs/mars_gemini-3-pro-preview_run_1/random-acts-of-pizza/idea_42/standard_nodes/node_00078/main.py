import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import (
    config,
    utils,
    features_text,
    features_meta,
    dataset,
    model_mlp,
    model_rf,
    trainer,
)


def run():
    # 1. Initialization
    print("Initializing Run...")
    utils.set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    # We load dataframes here primarily for labels and failure analysis matching
    print("Loading dataframes...")
    train_df, val_df, test_df = utils.load_data(
        return_val=True, parse_list_cols=["requester_subreddits_at_request"]
    )

    # Extract labels for evaluation
    y_val = val_df["requester_received_pizza"].astype(int).values
    y_train = train_df["requester_received_pizza"].astype(int).values

    # 3. Stream A: Random Forest Pipeline
    print("\n" + "=" * 40)
    print("Stream A: Consistency-Augmented Top-K Random Forest")
    print("=" * 40)

    # Assemble Features
    X_train_rf, X_val_rf, X_test_rf = model_rf.assemble_rf_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Train Model
    rf_model = model_rf.train_rf(X_train_rf, y_train, X_val_rf, y_val)

    # Validation Inference
    rf_val_probs = model_rf.predict_rf(rf_model, X_val_rf)

    # 4. Stream B: Persona-Aware Skip-Gated MLP Pipeline
    print("\n" + "=" * 40)
    print("Stream B: Persona-Aware Skip-Gated MLP")
    print("=" * 40)

    # Create DataLoaders
    # Note: This handles feature generation internally if not cached
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        load_cached_data=True, batch_size=config.MLP_BATCH_SIZE
    )

    # Determine Metadata Dimension dynamically
    sample_batch, _ = next(iter(train_loader))
    meta_dim = sample_batch["dense_metadata"].shape[1]
    print(f"Detected Dense Metadata Dimension: {meta_dim}")

    # Train Model
    mlp_model = trainer.run_training(
        train_loader,
        val_loader,
        meta_dim=meta_dim,
        device=device,
        epochs=config.MLP_EPOCHS,
    )

    # Validation Inference
    mlp_val_probs = model_mlp.predict_mlp(mlp_model, val_loader, device)

    # 5. Ensemble & Evaluation
    print("\n" + "=" * 40)
    print("Ensemble Evaluation")
    print("=" * 40)

    # Ensure lengths match (safety check)
    min_len = min(len(rf_val_probs), len(mlp_val_probs), len(y_val))
    if len(rf_val_probs) != min_len or len(mlp_val_probs) != min_len:
        print(
            f"Warning: Length mismatch. RF: {len(rf_val_probs)}, MLP: {len(mlp_val_probs)}, GT: {len(y_val)}"
        )
        rf_val_probs = rf_val_probs[:min_len]
        mlp_val_probs = mlp_val_probs[:min_len]
        y_val = y_val[:min_len]
        val_df = val_df.iloc[:min_len]

    # Weighted Average
    ensemble_val_probs = (config.WEIGHT_RF * rf_val_probs) + (
        config.WEIGHT_MLP * mlp_val_probs
    )

    # Calculate Metric
    final_val_auc = roc_auc_score(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    # Calculate absolute error
    errors = np.abs(y_val - ensemble_val_probs)

    # Prepare DataFrame for correlation
    analysis_df = val_df.copy()
    analysis_df["prediction_error"] = errors

    # Select numeric features
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["prediction_error", "requester_received_pizza"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Compute correlations
    correlations = {}
    for col in feature_cols:
        # Skip columns with all NaNs or constant values
        if analysis_df[col].nunique() <= 1 or analysis_df[col].isnull().all():
            continue

        # Handle NaNs for correlation
        series = analysis_df[col].fillna(analysis_df[col].median())
        corr = np.corrcoef(series, analysis_df["prediction_error"])[0, 1]

        if not np.isnan(corr):
            correlations[col] = corr

    # Print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name:<50}: {val:.4f}")

    # 7. Submission Generation
    threshold = 0.7135451153926904

    if final_val_auc > threshold:
        print(
            f"\nValidation AUC ({final_val_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set
        rf_test_probs = model_rf.predict_rf(rf_model, X_test_rf)
        mlp_test_probs = model_mlp.predict_mlp(mlp_model, test_loader, device)

        # Ensemble
        ensemble_test_probs = (config.WEIGHT_RF * rf_test_probs) + (
            config.WEIGHT_MLP * mlp_test_probs
        )

        # Create Submission File
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": ensemble_test_probs,
            }
        )

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({final_val_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
