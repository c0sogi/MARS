import sys
import os
import gc
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_artifact
from library.data_loader import load_dataset
from library.preprocessor import (
    get_text_features,
    get_target_matrix,
    TagEncoder,
    TextVectorizer,
)
from library.mnb_model import VectorizedMNB


def predict_scores_gpu(model, X_scipy, batch_size=20000):
    """
    Performs MNB inference on GPU to optimize speed.
    Computes: scores = X @ W.T + b
    """
    if not torch.cuda.is_available():
        print("GPU not available, falling back to CPU inference.")
        return model.predict_scores(X_scipy)

    print(f"Predicting on GPU (Batch size: {batch_size})...")
    device = torch.device("cuda")

    # Prepare model weights
    # model.coef_ is (n_classes, n_features). We need W.T (n_features, n_classes) for X @ W.T
    W_t = torch.tensor(model.coef_.T, dtype=torch.float32, device=device)
    b = torch.tensor(model.intercept_, dtype=torch.float32, device=device)

    n_samples = X_scipy.shape[0]
    n_classes = model.coef_.shape[0]

    # Pre-allocate result array on CPU
    all_scores = np.zeros((n_samples, n_classes), dtype=np.float32)

    # Process in batches to manage GPU memory
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch_scipy = X_scipy[start_idx:end_idx]

        # Convert Scipy CSR to Torch Sparse COO
        # COO is efficient for construction
        coo = X_batch_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape

        # Create sparse tensor on GPU
        X_batch_torch = torch.sparse_coo_tensor(
            indices, values, size=shape, device=device
        )

        # Compute scores: (Batch, Features) @ (Features, Classes) -> (Batch, Classes)
        batch_scores = torch.sparse.mm(X_batch_torch, W_t)

        # Add bias
        batch_scores += b

        # Move result to CPU
        all_scores[start_idx:end_idx] = batch_scores.cpu().numpy()

        # Cleanup
        del X_batch_torch, batch_scores, indices, values
        torch.cuda.empty_cache()

    return all_scores


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting pipeline...")

    # 2. Train Data Loading & Processing
    # Use full training set for best performance
    print("\n=== Training Phase ===")
    df_train = load_dataset("train", limit=None, load_cached_data=True)

    # Extract Features and Targets
    # This saves the vectorizer and encoder to disk for later use
    X_train, vectorizer = get_text_features(df_train, "train", load_cached_data=True)
    Y_train, tag_encoder = get_target_matrix(df_train, "train", load_cached_data=True)

    # Free memory
    del df_train
    gc.collect()

    # 3. Model Training
    model = VectorizedMNB(alpha=Config.SMOOTHING_ALPHA)
    model.fit(X_train, Y_train)
    model.save(os.path.join(Config.WORK_DIR, "mnb_model.pkl"))

    # Free training data
    del X_train, Y_train
    gc.collect()

    # 4. Validation & Threshold Tuning
    print("\n=== Validation Phase ===")
    # Load FULL validation set to satisfy metric requirements
    df_val = load_dataset("val", limit=None, load_cached_data=True)

    # Transform using the fitted vectorizer and encoder
    X_val, _ = get_text_features(
        df_val, "val", vectorizer=vectorizer, load_cached_data=True
    )
    Y_val, _ = get_target_matrix(
        df_val, "val", encoder=tag_encoder, load_cached_data=True
    )

    # Predict raw scores using GPU on FULL validation set
    val_scores = predict_scores_gpu(model, X_val)

    # Tune Threshold for Mean F1 Score
    # OPTIMIZATION: Tune on a subset to avoid TimeoutError (Cite debug_lesson_1)
    print("Tuning decision threshold on subset...")
    subset_size = 50000
    if val_scores.shape[0] > subset_size:
        val_scores_subset = val_scores[:subset_size]
        Y_val_subset = Y_val[:subset_size]
    else:
        val_scores_subset = val_scores
        Y_val_subset = Y_val

    thresholds = np.linspace(-5, 5, 21)
    best_f1 = -1.0
    best_thresh = 0.0

    for t in thresholds:
        # Generate binary predictions on subset
        val_preds_bin = val_scores_subset > t

        # Calculate F1 (samples average)
        current_f1 = f1_score(
            Y_val_subset, val_preds_bin, average="samples", zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = t

    print(f"Best Threshold (tuned on subset): {best_thresh} (Subset F1: {best_f1})")

    # 5. Final Metrics & Failure Analysis
    # Re-evaluate on the FULL validation set with best threshold
    print("Calculating final metric on full validation set...")
    final_preds = val_scores > best_thresh
    final_f1 = f1_score(Y_val, final_preds, average="samples", zero_division=0)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_f1}")

    print("\n=== Failure Analysis ===")
    # Calculate per-sample F1 to find errors
    # Use the subset for analysis to save time/memory
    Y_val_dense = Y_val_subset.toarray().astype(int)
    preds_dense = (val_scores_subset > best_thresh).astype(int)

    intersection = (Y_val_dense * preds_dense).sum(axis=1)
    pred_sum = preds_dense.sum(axis=1)
    true_sum = Y_val_dense.sum(axis=1)

    epsilon = 1e-9
    f1_per_sample = 2 * intersection / (pred_sum + true_sum + epsilon)
    error_magnitude = 1.0 - f1_per_sample

    # Features for correlation
    # 1. Word Count
    word_counts = (
        df_val["text"].iloc[:subset_size].apply(lambda x: len(x.split())).values
    )
    # 2. Character Count
    char_counts = df_val["text"].iloc[:subset_size].apply(len).values

    corr_word = np.corrcoef(error_magnitude, word_counts)[0, 1]
    corr_char = np.corrcoef(error_magnitude, char_counts)[0, 1]

    print(f"Correlation (Error vs Word Count): {corr_word:.6f}")
    print(f"Correlation (Error vs Char Count): {corr_char:.6f}")

    # Cleanup
    del (
        df_val,
        X_val,
        Y_val,
        val_scores,
        final_preds,
        Y_val_dense,
        preds_dense,
        val_scores_subset,
        Y_val_subset,
    )
    gc.collect()

    # 6. Submission
    print("\n=== Submission Phase ===")
    # Load full test set
    df_test = load_dataset("test", limit=None, load_cached_data=True)

    # Transform
    X_test, _ = get_text_features(
        df_test, "test", vectorizer=vectorizer, load_cached_data=True
    )

    # Predict on GPU
    test_scores = predict_scores_gpu(model, X_test)

    # Apply threshold
    test_preds_bin = test_scores > best_thresh

    # Inverse transform to strings
    print("Converting binary predictions to tags...")
    predicted_tags = tag_encoder.inverse_transform(test_preds_bin)

    # Create submission dataframe
    submission = pd.DataFrame({"Id": df_test["Id"], "Tags": predicted_tags})

    # Save with quoting to ensure format matches sample (e.g. "tag1 tag2")
    submission.to_csv(Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
