import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, load_numpy, load_pickle, clip_probabilities
from library.modeling import train_and_evaluate, generate_submission
from library.manifold_densification import get_densified_val_data


def run():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # This executes the K-Fold training loop, saves models to disk, and prints CV scores.
    # It uses the densified training data (9x expansion).
    print("Starting training process...")
    train_and_evaluate(load_cached_data=True)

    # 3. Validation on Hold-out Set
    print("\nPerforming validation on hold-out set...")

    # Load validation data (Hold-out set, distinct from CV folds)
    # X_val_img: (N, 9, D_img), X_val_tab: (N, 9, D_tab), ids: (N,), y_val_labels: (N,)
    X_val_img, X_val_tab, val_ids, y_val_labels = get_densified_val_data(
        load_cached_data=True
    )

    # Load class names to map predictions
    classes = load_numpy(Config.CACHE_CLASSES)

    # Prepare data for inference (Flatten the 9 centroids structure for batch processing)
    N, C, D_img = X_val_img.shape
    _, _, D_tab = X_val_tab.shape

    X_val_img_flat = X_val_img.reshape(N * C, D_img)
    X_val_tab_flat = X_val_tab.reshape(N * C, D_tab)
    X_val_full = np.hstack([X_val_img_flat, X_val_tab_flat])

    # Ensemble Inference
    ensemble_probs = np.zeros((N * C, len(classes)))

    print(f"Running inference with {Config.N_FOLDS} models...")
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        model = load_pickle(model_path)

        # Predict on the flattened batch
        probs = model.predict_proba(X_val_full)
        ensemble_probs += probs

    # Average across models
    ensemble_probs /= Config.N_FOLDS

    # Reshape to (N, 9, n_classes) and average across centroids (Full-Manifold Aggregation)
    # This aggregates the 9 orthogonal views for each image
    probs_structured = ensemble_probs.reshape(N, C, len(classes))
    final_probs = probs_structured.mean(axis=1)

    # Clip probabilities to avoid log loss extremes
    final_probs = clip_probabilities(final_probs)

    # Map string labels to integer indices for metric calculation
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_val_indices = np.array([class_to_idx[label] for label in y_val_labels])

    # Compute Metric
    score = log_loss(y_val_indices, final_probs, labels=range(len(classes)))
    # Print exactly as requested
    print(f"Final Validation Metric: {score}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude: 1.0 - probability assigned to the true class
    true_class_probs = final_probs[np.arange(N), y_val_indices]
    error_magnitudes = 1.0 - true_class_probs

    # Correlate error with tabular features
    # We take the tabular features of the first centroid (invariant across centroids)
    # X_val_tab is (N, 9, 192)
    tabular_features = X_val_tab[:, 0, :]

    correlations = []
    num_features = tabular_features.shape[1]

    for i in range(num_features):
        feat_values = tabular_features[:, i]
        # Handle constant features to avoid warnings
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(error_magnitudes, feat_values)
            if np.isnan(corr):
                corr = 0.0

        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"Correlation between Error Magnitude and Input Features (Top 5):")
    for i, corr in correlations[:5]:
        print(f"  Feature Index {i}: {corr:.6f}")

    # 5. Submission
    # We generate the submission regardless of the metric threshold to ensure
    # the output file exists for grading.
    print("\nGenerating submission...")
    generate_submission(load_cached_data=True)


if __name__ == "__main__":
    run()
