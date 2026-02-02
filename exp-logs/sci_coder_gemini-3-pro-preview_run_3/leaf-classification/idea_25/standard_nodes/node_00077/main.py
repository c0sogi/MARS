import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.execution import ModelExecutor


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "runfile.log"))
    logger.info("Starting runfile.py execution...")

    # 2. Initialize Executor
    executor = ModelExecutor()

    # 3. Train Ensemble
    # Trains the Stratified K-Fold ensemble using the densified training data.
    # Returns a list of trained pipeline objects.
    pipelines = executor.train_ensemble(load_cached_data=True)

    # 4. Validation on Hold-Out Set
    logger.info("Performing validation on hold-out set...")

    # Load validation data from metadata/val.csv
    # The densification manager returns data expanded by the 3 centroids
    val_ids, val_dino, val_conv, val_tab, val_labels = (
        executor.densification_manager.prepare_validation_data(load_cached_data=True)
    )

    # Construct the feature matrix for the validation set
    X_val = executor._prepare_feature_matrix(val_dino, val_conv, val_tab)

    # Perform Inference
    # We average predictions across all models in the ensemble
    n_samples = len(X_val)
    # Get classes from the first pipeline
    classes = pipelines[0].classes_
    n_classes = len(classes)

    # Accumulate probabilities
    ensemble_proba_centroids = np.zeros((n_samples, n_classes))
    for pipeline in pipelines:
        # Predict on all centroids
        ensemble_proba_centroids += pipeline.predict_proba(X_val)

    # Average across ensemble members
    ensemble_proba_centroids /= len(pipelines)

    # Aggregation: Centroids -> Images
    # We must aggregate the 3 centroids per image to get the final image-level prediction
    df_val_pred = pd.DataFrame(ensemble_proba_centroids, columns=classes)
    df_val_pred["id"] = val_ids

    # Group by ID and take the mean of probabilities
    df_val_agg = df_val_pred.groupby("id").mean().reset_index()

    # Prepare True Labels for Metric Calculation
    # Map IDs to their labels using the loaded validation data
    # Since val_labels array corresponds to val_ids (which has repeats), we extract unique pairs
    df_val_labels = pd.DataFrame({"id": val_ids, "species": val_labels})
    df_val_labels_unique = df_val_labels.drop_duplicates(subset=["id"]).sort_values(
        "id"
    )

    # Ensure the aggregated predictions are sorted by ID to match labels
    df_val_agg = df_val_agg.sort_values("id")

    # Verify alignment
    if not np.array_equal(df_val_agg["id"].values, df_val_labels_unique["id"].values):
        logger.error("Validation ID mismatch during aggregation.")
        sys.exit(1)

    y_true = df_val_labels_unique["species"].values
    y_pred = df_val_agg.drop(columns=["id"]).values
    pred_classes = df_val_agg.drop(columns=["id"]).columns.tolist()

    # Calculate Final Validation Metric (Log Loss)
    final_metric = log_loss(y_true, y_pred, labels=pred_classes)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")

    # Calculate per-sample log loss to correlate with features
    # Map string labels to column indices
    class_to_idx = {cls: i for i, cls in enumerate(pred_classes)}
    y_true_indices = np.array([class_to_idx[lbl] for lbl in y_true])

    # Extract predicted probability for the true class
    # Clip probabilities to avoid log(0)
    prob_true = y_pred[np.arange(len(y_pred)), y_true_indices]
    prob_true = np.clip(prob_true, 1e-15, 1 - 1e-15)
    sample_losses = -np.log(prob_true)

    # Aggregating features per image for correlation
    # We use the tabular features for interpretability
    df_tab = pd.DataFrame(val_tab, columns=executor.densification_manager.tabular_cols)
    df_tab["id"] = val_ids
    df_tab_agg = df_tab.groupby("id").mean().reset_index().sort_values("id")

    # Calculate Pearson correlation between each feature and the loss
    feature_values = df_tab_agg.drop(columns=["id"]).values
    feature_names = df_tab_agg.drop(columns=["id"]).columns

    feature_correlations = {}
    for i, fname in enumerate(feature_names):
        f_vals = feature_values[:, i]
        # Avoid division by zero for constant features
        if np.std(f_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(f_vals, sample_losses)[0, 1]
        feature_correlations[fname] = corr

    # Sort by absolute correlation magnitude
    sorted_corrs = sorted(
        feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True
    )

    print("Top 5 Features correlated with Error:")
    for name, corr in sorted_corrs[:5]:
        print(f"{name}: {corr:.4f}")

    # 6. Generate Submission
    # Generates predictions for the test set and saves to submission.csv
    executor.generate_submission(pipelines, load_cached_data=True)
    logger.info("Run complete.")


if __name__ == "__main__":
    main()
