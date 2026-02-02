import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library import utils
from library import execution_engine
from library import data_manager


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)

    # 2. Train Ensemble
    # This function orchestrates the entire training pipeline including:
    # - Feature extraction (with caching)
    # - Manifold densification (3 centroids per image)
    # - Stratified K-Fold training of the LDA ensemble
    execution_engine.train_ensemble(load_cached_data=True)

    # 3. Validation Assessment
    # We perform inference on the hold-out validation set defined in metadata/val.csv
    print("Performing Validation Assessment...")

    dm = data_manager.LeafDataManager()
    val_data = dm.get_dataset("val", load_cached_data=True)

    # Prepare Validation Data
    # Concatenate features: [DINO | CONV | TABULAR]
    X_val = np.hstack([val_data["dino"], val_data["conv"], val_data["tabular"]])
    y_val_densified = val_data["y"]
    ids_val_densified = val_data["ids"]

    # Recover unique IDs and Labels (data is densified 3x)
    unique_val_ids = ids_val_densified[::3]
    unique_y_val = y_val_densified[::3]

    n_samples = len(unique_val_ids)

    # Load Classes
    models_dir = os.path.join(config.WORKING_DIR, "models")
    classes_path = os.path.join(models_dir, "classes.pkl")
    if not os.path.exists(classes_path):
        raise FileNotFoundError("Classes file not found. Training might have failed.")

    classes = utils.load_pickle(classes_path)
    n_classes = len(classes)

    # Ensemble Inference
    ensemble_probs = np.zeros((n_samples, n_classes))

    for fold in range(config.N_FOLDS):
        model_path = os.path.join(models_dir, f"pipeline_fold_{fold}.pkl")
        if not os.path.exists(model_path):
            continue

        pipeline = utils.load_pickle(model_path)

        # Predict on densified validation data (3N samples)
        probs_densified = pipeline.predict_proba(X_val)

        # Reshape to (N, 3, C) and average across centroids
        probs_reshaped = probs_densified.reshape(n_samples, 3, n_classes)
        probs_agg = np.mean(probs_reshaped, axis=1)

        ensemble_probs += probs_agg

    # Average across folds
    final_val_probs = ensemble_probs / config.N_FOLDS
    final_val_probs = utils.clip_probabilities(final_val_probs)

    # Calculate Metric
    # Map string labels to indices
    class_map = {c: i for i, c in enumerate(classes)}
    y_val_indices = np.array([class_map[label] for label in unique_y_val])

    val_score = log_loss(y_val_indices, final_val_probs, labels=list(range(n_classes)))
    print(f"Final Validation Metric: {val_score}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate per-sample loss: -log(p_true)
    # Advanced indexing to select the probability of the true class for each sample
    true_probs = final_val_probs[np.arange(n_samples), y_val_indices]
    sample_losses = -np.log(true_probs)

    # Correlate with Tabular Features
    # val_data["tabular"] is (3N, 192). We take every 3rd row to match unique samples.
    val_tabular_unique = val_data["tabular"][::3]
    feature_names = dm.tabular_cols

    correlations = []
    for i, name in enumerate(feature_names):
        feat_values = val_tabular_unique[:, i]
        # Avoid correlation with constant features
        if np.std(feat_values) > 1e-9:
            corr, _ = pearsonr(sample_losses, feat_values)
            correlations.append((name, corr))
        else:
            correlations.append((name, 0.0))

    # Sort by magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # 5. Submission
    # Generate submission unconditionally to ensure file presence
    execution_engine.predict_submission(load_cached_data=True)


if __name__ == "__main__":
    main()
