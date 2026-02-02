import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.metrics import accuracy_score
from library import config, data, ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting End-to-End Self-Training Pipeline...")

    # --- 1. Data Loading ---
    print("Loading and Preprocessing Data...")
    # Load data using library functions (utilizes caching)
    df_train = data.load_and_preprocess("train", load_cached_data=True)
    df_val = data.load_and_preprocess("val", load_cached_data=True)
    df_test = data.load_and_preprocess("test", load_cached_data=True)

    # Separate Features and Targets
    X_train, y_train = data.get_X_y(df_train)
    X_val, y_val = data.get_X_y(df_val)
    X_test, _ = data.get_X_y(df_test)

    # Store Test IDs for submission
    test_ids = df_test[config.ID_COL]

    print(
        f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}"
    )

    # Create a combined inference set (Val + Test) to get predictions for both in one pass
    # We track the split index to separate them later
    n_val = len(X_val)
    X_inference = pd.concat([X_val, X_test], axis=0, ignore_index=True)

    # Initialize Pipeline
    pipeline = ensemble.EnsemblePipeline()

    # --- 2. Stage 1: Teacher Training ---
    print("\n=== Stage 1: Teacher Training (Original Data) ===")
    # Train on original data, predict on combined inference set
    # oof_s1 is for the training set (not used here), inf_probs_s1 is for X_inference
    _, inf_probs_s1 = pipeline.run_cv_training(X_train, y_train, X_inference)

    # Split predictions back into Val and Test
    val_probs_s1 = inf_probs_s1[:n_val]
    test_probs_s1 = inf_probs_s1[n_val:]

    # Calculate Stage 1 Validation Score
    val_preds_s1 = np.argmax(val_probs_s1, axis=1)
    acc_s1 = accuracy_score(y_val, val_preds_s1)
    print(f"Stage 1 Validation Accuracy: {acc_s1}")

    # --- 3. Pseudo-Labeling ---
    print("\n=== Generating Pseudo-Labels ===")
    # Augment training data with high-confidence test predictions
    X_train_aug, y_train_aug = pipeline.generate_augmented_train_set(
        X_train, y_train, X_test, test_probs_s1
    )

    print(f"Augmented Train Shape: {X_train_aug.shape}")

    # --- 4. Stage 2: Student Training ---
    print("\n=== Stage 2: Student Training (Augmented Data) ===")
    # Train on augmented data.
    # IMPORTANT: Pass n_original_samples to restrict validation to original data only.
    _, inf_probs_s2 = pipeline.run_cv_training(
        X_train_aug, y_train_aug, X_inference, n_original_samples=len(X_train)
    )

    # Split predictions
    val_probs_s2 = inf_probs_s2[:n_val]
    test_probs_s2 = inf_probs_s2[n_val:]

    # --- 5. Final Evaluation ---
    val_preds_s2 = np.argmax(val_probs_s2, axis=1)
    final_acc = accuracy_score(y_val, val_preds_s2)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude: 1.0 - Probability assigned to the True Class
    # y_val contains mapped classes (0..5). val_probs_s2 is (N, 6).
    row_indices = np.arange(len(y_val))

    # Extract probability of the true class
    # y_val is a Series, convert to numpy for indexing
    y_val_np = y_val.values
    true_class_probs = val_probs_s2[row_indices, y_val_np]

    # Error magnitude (0.0 = perfect confidence in correct class, 1.0 = zero confidence)
    error_magnitude = 1.0 - true_class_probs

    # Calculate correlation with features
    correlations = []
    # Analyze numerical features
    num_cols = X_val.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        try:
            # Check for constant columns to avoid warnings
            if X_val[col].std() == 0:
                continue

            # Pearson correlation
            corr = np.corrcoef(X_val[col], error_magnitude)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))
        except Exception:
            continue

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Error Magnitude:")
    print(f"{'Feature':<40} {'Correlation':<10}")
    print("-" * 50)
    for name, corr in correlations[:10]:
        print(f"{name:<40} {corr:.4f}")

    # --- 7. Submission ---
    THRESHOLD = 0.9619111111111112

    if final_acc > THRESHOLD:
        print(f"\nValidation Metric ({final_acc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        pipeline.save_submission(test_ids, test_probs_s2)
    else:
        print(
            f"\nValidation Metric ({final_acc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
