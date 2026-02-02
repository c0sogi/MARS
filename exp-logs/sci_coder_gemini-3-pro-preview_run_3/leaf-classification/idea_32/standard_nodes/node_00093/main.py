import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.engine import Engine


def main():
    # 1. Setup and Initialization
    seed_everything(Config.SEED)

    # Initialize the orchestration engine
    engine = Engine()

    # 2. Training Phase
    # Performs Stratified K-Fold CV on the training set
    # Populates engine.models with the trained ensemble
    engine.train_folds()

    # 3. Hold-out Validation Phase
    print("\nPerforming Hold-out Validation on metadata/val.csv...")

    # Load densified validation data (3 centroids per image)
    # d_img: (N*3, D_visual), d_tab: (N*3, D_tabular)
    d_img, d_tab, d_ids, d_labels = engine.processor.get_val_data(load_cached=True)

    # Concatenate to form full feature matrix: [DINO | ConvNeXt | Tabular]
    X_val = np.concatenate([d_img, d_tab], axis=1)

    # Determine number of unique images
    # Data is structured as blocks of 3: [img1_c1, img1_c2, img1_c3, img2_c1, ...]
    n_samples = len(d_ids) // 3

    # Initialize ensemble probability accumulator
    ensemble_probs = np.zeros((len(d_ids), len(engine.class_names)))

    # Inference with the trained ensemble
    for i, clf in enumerate(engine.models):
        probs = clf.predict_proba(X_val)
        ensemble_probs += probs

    # Average across models
    ensemble_probs /= len(engine.models)

    # Aggregate Centroids: Average probabilities of the 3 views per image
    # Reshape to (N_images, 3, N_classes) -> Mean -> (N_images, N_classes)
    probs_reshaped = ensemble_probs.reshape(n_samples, 3, -1)
    probs_avg = np.mean(probs_reshaped, axis=1)

    # Extract true labels (stride of 3 to get one per image)
    y_true = d_labels[::3]

    # Compute Final Metric
    metric = log_loss(y_true, probs_avg, labels=engine.class_names)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample log loss
    # Map string labels to indices
    class_to_idx = {cls: i for i, cls in enumerate(engine.class_names)}
    y_indices = np.array([class_to_idx[y] for y in y_true])

    # Extract probability assigned to the true class
    # Clip to avoid log(0)
    probs_clipped = clip_probabilities(probs_avg)
    true_class_probs = probs_clipped[np.arange(len(y_indices)), y_indices]

    # Loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Correlate error with input feature characteristics
    # We aggregate features back to image level for correlation
    X_reshaped = X_val.reshape(n_samples, 3, -1)
    X_avg = np.mean(X_reshaped, axis=1)

    # Feature Slices based on Config:
    # DINO: 0-1024
    # ConvNeXt: 1024-2560 (1024 + 1536)
    # Tabular: 2560-2752 (2560 + 192)
    dino_dim = 1024
    conv_dim = 1536

    dino_mean = np.mean(X_avg[:, :dino_dim], axis=1)
    conv_mean = np.mean(X_avg[:, dino_dim : dino_dim + conv_dim], axis=1)
    tab_mean = np.mean(X_avg[:, dino_dim + conv_dim :], axis=1)

    # Compute correlations
    corr_dino, _ = pearsonr(sample_losses, dino_mean)
    corr_conv, _ = pearsonr(sample_losses, conv_mean)
    corr_tab, _ = pearsonr(sample_losses, tab_mean)

    print(f"Correlation (Error vs DINO Mean Signal): {corr_dino:.4f}")
    print(f"Correlation (Error vs ConvNeXt Mean Signal): {corr_conv:.4f}")
    print(f"Correlation (Error vs Tabular Mean Signal): {corr_tab:.4f}")

    # 5. Submission
    # Generate submission file for the test set
    engine.generate_submission()


if __name__ == "__main__":
    main()
