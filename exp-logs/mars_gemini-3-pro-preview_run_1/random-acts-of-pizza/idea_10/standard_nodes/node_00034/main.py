import os
import numpy as np
import pandas as pd
import torch
import scipy.stats

from library import config, utils, feature_engineering, models


def run():
    # 1. Initialization
    print("Initializing workflow...")
    utils.set_seed(config.SEED)

    # 2. Feature Engineering
    print("Running Feature Pipeline...")
    pipeline = feature_engineering.FeaturePipeline()
    # load_cached_data=True attempts to load from ./working/idea_10/ if available
    # otherwise generates from scratch using ./metadata/ inputs
    train_data, val_data, test_data = pipeline.run(load_cached_data=True)

    # 3. Stream A: Random Forest
    print("\n=== Stream A: Topic-Augmented Random Forest ===")
    rf_model = models.StreamARandomForest()

    # Train
    rf_model.train(
        train_data["stream_a"]["X_tfidf"],
        train_data["stream_a"]["X_meta"],
        train_data["y"],
    )

    # Validate
    pred_a_val = rf_model.predict_proba(
        val_data["stream_a"]["X_tfidf"], val_data["stream_a"]["X_meta"]
    )

    # 4. Stream B: Context-Gated MLP
    print("\n=== Stream B: Context-Gated MLP ===")
    # Determine metadata dimension from the processed data
    meta_dim = train_data["stream_b"]["X_meta"].shape[1]

    mlp_trainer = models.MLPTrainer(meta_dim=meta_dim)

    # Train (Trainer handles DataLoaders and Early Stopping)
    mlp_trainer.fit(train_data, val_data)

    # Validate
    pred_b_val = mlp_trainer.predict_proba(val_data)

    # 5. Ensemble & Evaluation
    print("\n=== Ensemble Evaluation ===")
    # Simple Weighted Average (0.5 / 0.5) as per config
    w_a, w_b = config.ENSEMBLE_WEIGHTS
    final_pred_val = (w_a * pred_a_val) + (w_b * pred_b_val)

    # Compute Metric
    val_auc = utils.compute_score(val_data["y"], final_pred_val)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    y_val = val_data["y"]
    errors = np.abs(y_val - final_pred_val)

    # Correlate errors with metadata features (using Stream B's normalized metadata)
    # X_meta is a numpy array. We don't have column names easily accessible from the pipeline output
    # without reconstructing them, so we'll use indices.
    X_meta_val = val_data["stream_b"]["X_meta"]

    correlations = []
    for i in range(X_meta_val.shape[1]):
        feature_col = X_meta_val[:, i]
        # Handle potential constant columns to avoid warnings
        if np.std(feature_col) > 1e-9:
            corr, _ = scipy.stats.pearsonr(feature_col, errors)
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Metadata Features correlated with Prediction Error:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    # 7. Submission
    threshold = 0.6959737721862433
    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")

        # Inference on Test Set
        print("Predicting Stream A (Test)...")
        pred_a_test = rf_model.predict_proba(
            test_data["stream_a"]["X_tfidf"], test_data["stream_a"]["X_meta"]
        )

        print("Predicting Stream B (Test)...")
        pred_b_test = mlp_trainer.predict_proba(test_data)

        # Ensemble
        final_pred_test = (w_a * pred_a_test) + (w_b * pred_b_test)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_data["ids"],
                "requester_received_pizza": final_pred_test,
            }
        )

        # Save
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation metric {val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
