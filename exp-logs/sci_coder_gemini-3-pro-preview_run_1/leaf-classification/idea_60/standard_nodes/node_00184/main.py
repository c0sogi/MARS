import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from library import config, data, model, utils

# Set seeds for reproducibility
utils.set_seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)


def predict_proba_gpu(W, b, X):
    """
    Performs linear inference and softmax on the GPU (if available).

    Args:
        W (np.ndarray): Weights matrix (n_classes, n_features).
        b (np.ndarray): Bias vector (n_classes,).
        X (np.ndarray): Input features (n_samples, n_features).

    Returns:
        np.ndarray: Class probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors (ensure float64 for precision)
    # Using torch.double for float64
    W_t = torch.tensor(W, dtype=torch.double).to(device)
    b_t = torch.tensor(b, dtype=torch.double).to(device)
    X_t = torch.tensor(X, dtype=torch.double).to(device)

    # Inference: Z = X @ W.T + b
    # X: (N, D), W: (K, D), b: (K,)
    # Result: (N, K)
    with torch.no_grad():
        logits = X_t @ W_t.T + b_t
        probs = F.softmax(logits, dim=1)

    return probs.cpu().numpy()


def perform_failure_analysis(X_val, y_val, probs, feature_names=None):
    """
    Analyzes which features correlate with high prediction error.
    """
    utils.Logger.info("Performing Failure Analysis...")

    # 1. Calculate per-sample Log Loss
    # Gather probability assigned to the true class
    # y_val are indices 0..K-1
    n_samples = len(y_val)
    true_class_probs = probs[np.arange(n_samples), y_val]

    # Clip for stability
    true_class_probs = np.maximum(true_class_probs, 1e-15)

    # Loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # 2. Correlate features with Loss
    # X_val is (N, D)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feat_values = X_val[:, i]
        # Handle constant features (though sanitizer should have removed them)
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, sample_losses)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # 3. Report Top Correlations
    # Sort by absolute correlation
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("\n--- Failure Analysis: Top Features Correlated with Error ---")
    for idx in top_indices:
        feat_name = f"Feature_{idx}"  # We don't have explicit names for the processed array easily available here
        print(f"Feature Index {idx}: Correlation = {correlations[idx]:.4f}")
    print("------------------------------------------------------------\n")


def main():
    utils.Logger.info("Starting Runfile Execution...")

    # 1. Data Preparation
    dm = data.LeafDataManager()
    # Load data (cached if available)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = dm.prepare_data(
        load_cached_data=True
    )

    utils.Logger.info(
        f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 2. Model Training
    utils.Logger.info("Training Sanitized OAS Discriminant...")
    clf = model.SanitizedOASDiscriminant()

    with utils.Timer("Model Training"):
        clf.fit(X_train, y_train)

    # 3. Validation Inference (GPU)
    utils.Logger.info("Running Validation Inference on GPU...")
    val_probs = predict_proba_gpu(clf.W_, clf.b_, X_val)

    # 4. Metric Calculation
    # Apply strict clipping and rescaling as per task description
    # "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)"
    val_probs_clipped = np.maximum(np.minimum(val_probs, 1 - 1e-15), 1e-15)
    # "rescaled prior to being scored (each row is divided by the row sum)"
    val_probs_normalized = val_probs_clipped / val_probs_clipped.sum(
        axis=1, keepdims=True
    )

    val_loss = log_loss(y_val, val_probs_normalized, labels=np.arange(len(classes)))

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    perform_failure_analysis(X_val, y_val, val_probs_normalized)

    # 6. Submission Generation
    # Threshold from task description
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        utils.Logger.info(
            f"Validation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set (GPU)
        test_probs = predict_proba_gpu(clf.W_, clf.b_, X_test)

        # Clip and Normalize
        test_probs_clipped = np.maximum(np.minimum(test_probs, 1 - 1e-15), 1e-15)
        # Note: We don't strictly need to normalize for submission as the scorer does it,
        # but it's good practice to submit valid probabilities.
        # We will submit the clipped values directly as requested by the format.

        # Create DataFrame
        submission_df = pd.DataFrame(test_probs_clipped, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save
        if not os.path.exists(config.SUBMISSION_DIR):
            os.makedirs(config.SUBMISSION_DIR)

        submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        utils.Logger.info(f"Submission saved to {config.SUBMISSION_FILE_PATH}")

    else:
        utils.Logger.info(
            f"Validation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
