import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.train import Trainer
from library.data_processing import DatasetManager


def main():
    # ==========================================
    # 1. Training & Submission Generation
    # ==========================================
    # The Trainer class handles:
    # - Feature Extraction (utilizing GPU and caching)
    # - Stratified K-Fold CV (Ensemble creation)
    # - Model Training (LDA with Manifold Densification)
    # - Submission Generation (Test-Time Aggregation on Test Set)

    print("Initializing Training Pipeline...")
    trainer = Trainer()
    trainer.run()

    # ==========================================
    # 2. Validation on Hold-Out Set
    # ==========================================
    # We evaluate the trained ensemble on the specific validation set defined in metadata/val.csv
    # to compute the Final Validation Metric and perform Failure Analysis.

    print("\n--- Validation & Failure Analysis ---")

    # Load validation metadata to get IDs and Labels
    val_meta_path = Config.VAL_CSV
    if not os.path.exists(val_meta_path):
        print(f"Error: Validation metadata not found at {val_meta_path}")
        return

    df_val_meta = pd.read_csv(val_meta_path)
    val_ids_target = df_val_meta["id"].values

    # Load all features (cached) via DatasetManager
    # The 'train' key in the loaded data contains combined train+val samples
    dm = DatasetManager()
    data = dm.load_data(load_cached_data=True)

    # The 'val' dictionary contains the isolated validation set
    if "val" not in data:
        print("Error: Validation data not found in loaded dataset.")
        return

    val_subset = data["val"]

    # Prepare densified data for inference
    # This converts N images into 3N samples (3 orthogonal centroids per image)
    # Returns X_val (3N, D), y_val (3N,), ids (3N,)
    X_val, y_val_expanded, ids_val_expanded = dm.prepare_training_set(val_subset)

    # Load Class Definitions (generated during training)
    classes_path = os.path.join(Config.WORKING_DIR, "models", "classes.pkl")
    if not os.path.exists(classes_path):
        print("Error: Classes file not found. Training may have failed.")
        return

    with open(classes_path, "rb") as f:
        classes = pickle.load(f)

    n_samples = len(val_subset["ids"])
    n_classes = len(classes)

    # Ensemble Prediction
    # Aggregate predictions from all K folds
    ensemble_probs = np.zeros((n_samples, n_classes))
    models_dir = os.path.join(Config.WORKING_DIR, "models")

    models_found = 0
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(models_dir, f"pipeline_fold_{fold}.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)

            # Predict on densified inputs (3 views per image)
            probs_expanded = pipeline.predict_proba(X_val)

            # Reshape (N, 3, C) and Average (Test-Time Aggregation)
            probs_reshaped = probs_expanded.reshape(n_samples, 3, n_classes)
            probs_mean = np.mean(probs_reshaped, axis=1)

            ensemble_probs += probs_mean
            models_found += 1

    if models_found > 0:
        ensemble_probs /= models_found
    else:
        print("Error: No trained models found.")
        return

    # Calculate Final Metric
    # Use the labels corresponding to the subset features
    y_true = val_subset["labels"]
    final_metric = log_loss(y_true, ensemble_probs, labels=classes)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 3. Failure Analysis
    # ==========================================
    # Calculate error magnitude: 1 - Probability(True Class)

    # Map string labels to integer indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_indices = np.array([class_to_idx[lbl] for lbl in y_true])

    # Extract probability assigned to the true class
    # shape: (N,)
    prob_true_class = ensemble_probs[np.arange(n_samples), y_true_indices]

    # Error magnitude
    error_magnitude = 1.0 - prob_true_class

    print(f"Mean Error Magnitude: {np.mean(error_magnitude):.6f}")

    # Correlate Error with Tabular Features
    # We check which input features correlate with higher error
    tabular_feats = val_subset["tabular"]  # (N, 192)

    correlations = []
    # Reconstruct feature names based on config/description
    # 0-63: margin, 64-127: shape, 128-191: texture
    feat_names = []
    for i in range(64):
        feat_names.append(f"margin_{i+1}")
    for i in range(64):
        feat_names.append(f"shape_{i+1}")
    for i in range(64):
        feat_names.append(f"texture_{i+1}")

    # Calculate Pearson correlation for each feature
    for i in range(tabular_feats.shape[1]):
        f_vals = tabular_feats[:, i]
        # Handle constant features to avoid warnings
        if np.std(f_vals) < 1e-12:
            corr = 0.0
        else:
            corr, _ = pearsonr(error_magnitude, f_vals)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get Top 5 correlations (absolute value)
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Feature Correlations with Error Magnitude:")
    for idx in top_indices:
        print(f"  {feat_names[idx]}: {correlations[idx]:.4f}")


if __name__ == "__main__":
    main()
