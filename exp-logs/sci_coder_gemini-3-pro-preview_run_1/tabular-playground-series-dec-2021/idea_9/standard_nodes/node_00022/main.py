import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.metrics import accuracy_score
from library import config, data, ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Homogeneous Ensemble Pipeline...")

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

    # --- 2. Ensemble Training ---
    print("\n=== Training Homogeneous Ensemble (Stratified K-Fold) ===")
    # Train on training data, predict on combined inference set
    # oof_probs is for the training set, inf_probs is for X_inference
    _, inf_probs = pipeline.run_cv_training(X_train, y_train, X_inference)

    # Split predictions back into Val and Test
    val_probs = inf_probs[:n_val]
    test_probs = inf_probs[n_val:]

    # --- 3. Final Evaluation ---
    val_preds = np.argmax(val_probs, axis=1)
    final_acc = accuracy_score(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # --- 4. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude: 1.0 - Probability assigned to the True Class
    # y_val contains mapped classes (0..5). val_probs is (N, 6).
    row_indices = np.arange(len(y_val))

    # Extract probability of the true class
    # y_val is a Series, convert to numpy for indexing
    y_val_np = y_val.values
    true_class_probs = val_probs[row_indices, y_val_np]

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

    # --- 5. Submission ---
    THRESHOLD = 0.9619111111111112

    if final_acc > THRESHOLD:
        print(f"\nValidation Metric ({final_acc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        pipeline.save_submission(test_ids, test_probs)
    else:
        print(
            f"\nValidation Metric ({final_acc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
