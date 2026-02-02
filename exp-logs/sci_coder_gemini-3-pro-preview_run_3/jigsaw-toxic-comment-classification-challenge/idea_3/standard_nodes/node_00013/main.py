import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(".")

# Import library modules
from library.config import Config
from library.trainer import run_training
from library.nbsvm_model import train_and_predict_nbsvm
from library.dataset import load_data_splits, get_nbsvm_features
from library.utils import calculate_roc_auc, seed_everything


def main():
    # --- 1. Configuration Setup ---
    # Modify Config for Fast Baseline execution within time limits
    # We restrict training to 1 epoch to ensure completion within the 2-hour limit on the provided hardware.
    # We increase the inference batch size to 64 to fully utilize the A100 GPU memory.
    print("Configuring parameters...")
    Config.EPOCHS = 1
    Config.VALID_BATCH_SIZE = 64

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --- 2. Neural Model Training (DeBERTa) ---
    print("\n=== Neural Model Pipeline ===")
    # run_training handles data loading, training, and inference internally.
    # It returns the raw probability predictions for validation and test sets.
    deb_val_preds, deb_test_preds = run_training()

    # --- 3. NBSVM Model Training ---
    print("\n=== NBSVM Pipeline ===")
    # Load data splits explicitly for NBSVM feature generation
    df_train, df_val, df_test = load_data_splits()

    # Generate/Load TF-IDF features
    # load_cached_data=True allows using pre-computed features if available in ./working
    X_train, X_val, X_test = get_nbsvm_features(
        df_train, df_val, df_test, load_cached_data=True
    )

    # Extract labels
    y_train = df_train[Config.LABEL_COLS].values
    y_val = df_val[Config.LABEL_COLS].values

    # Train NBSVM and get predictions
    nb_val_preds, nb_test_preds = train_and_predict_nbsvm(
        X_train, y_train, X_val, y_val, X_test
    )

    # --- 4. Ensemble Optimization ---
    print("\n=== Ensemble Optimization ===")
    # Optimize mixing weight alpha on validation set
    # P_final = alpha * P_deberta + (1 - alpha) * P_nbsvm

    best_score = -1.0
    best_alpha = 0.5

    # Search space: 0.0 to 1.0 with step 0.05
    alphas = np.linspace(0, 1, 21)

    for alpha in alphas:
        # Weighted average
        ens_preds = (alpha * deb_val_preds) + ((1 - alpha) * nb_val_preds)
        score = calculate_roc_auc(y_val, ens_preds)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"Optimal Alpha (DeBERTa Weight): {best_alpha:.2f}")
    print(f"Best Validation Score (Optimization): {best_score}")

    # --- 5. Final Validation & Metric ---
    # Compute final validation predictions using best alpha
    final_val_preds = (best_alpha * deb_val_preds) + ((1 - best_alpha) * nb_val_preds)

    # Calculate final metric
    final_metric = calculate_roc_auc(y_val, final_val_preds)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude per sample
    # We use Mean Absolute Error (MAE) averaged across the 6 labels
    # y_val is (N, 6), final_val_preds is (N, 6)
    error_per_sample = np.mean(np.abs(y_val - final_val_preds), axis=1)

    # Feature: Word Count
    # Handle NaNs just in case, though dataset loader handles it
    texts = df_val["comment_text"].fillna("").astype(str)
    word_counts = texts.apply(lambda x: len(x.split())).values

    # Calculate Correlation
    corr, p_value = pearsonr(error_per_sample, word_counts)
    print(f"Correlation between Error Magnitude and Word Count: {corr}")

    # --- 7. Submission Generation ---
    THRESHOLD = 0.9930242958008497

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Compute final test predictions
        final_test_preds = (best_alpha * deb_test_preds) + (
            (1 - best_alpha) * nb_test_preds
        )

        # Prepare submission DataFrame
        # df_test contains 'id' from metadata
        submission = pd.DataFrame()
        submission["id"] = df_test["id"]

        # Add prediction columns
        for i, col in enumerate(Config.LABEL_COLS):
            submission[col] = final_test_preds[:, i]

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
