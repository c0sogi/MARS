import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add the current directory to path to ensure library imports work if not implicitly handled
sys.path.append(os.getcwd())

from library.utils import set_seed, load_data, save_submission
from library.execution import run_nblr, run_neural, optimize_ensemble


def main():
    # 1. Setup
    set_seed(42)
    warnings.filterwarnings("ignore")

    # 2. Run Statistical Stream (NBSVM)
    # We use the full dataset (max_samples=None) as it is small enough for rapid training
    print("Executing Statistical Stream...")
    nb_val_preds, nb_test_preds = run_nblr(
        load_cached_data=True, max_samples=None, C=1.0, dual=True
    )

    # 3. Run Neural Stream (RoBERTa)
    # 3 epochs is sufficient for fine-tuning on this dataset size
    print("Executing Neural Stream...")
    neural_val_preds, neural_test_preds = run_neural(
        load_cached_data=True, max_samples=None, epochs=3, batch_size=16, lr=2e-5
    )

    # 4. Load Validation Labels for Optimization and Analysis
    val_df = load_data("val", load_cached_data=True)
    y_val = val_df["Insult"].values

    # 5. Optimize Ensemble
    # Finds the best weight 'w' such that: w * nb + (1-w) * neural
    best_w, final_auc = optimize_ensemble(y_val, nb_val_preds, neural_val_preds)

    # 6. Report Validation Metric (Strict Format)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate final ensemble predictions on validation set
    final_val_preds = best_w * nb_val_preds + (1 - best_w) * neural_val_preds

    # Calculate error magnitude
    errors = np.abs(y_val - final_val_preds)

    # Feature 1: Character Length
    # Ensure comment is string to avoid len() errors on potential NaNs (though load_data cleans this)
    val_df["char_len"] = val_df["Comment"].astype(str).apply(len)
    corr_char = np.corrcoef(val_df["char_len"], errors)[0, 1]
    print(f"Correlation between Error and Character Length: {corr_char}")

    # Feature 2: Word Count
    val_df["word_count"] = val_df["Comment"].astype(str).apply(lambda x: len(x.split()))
    corr_word = np.corrcoef(val_df["word_count"], errors)[0, 1]
    print(f"Correlation between Error and Word Count: {corr_word}")

    # 8. Conditional Submission
    THRESHOLD = 0.8839408866995074

    if final_auc > THRESHOLD:
        print(
            f"\nPerformance check passed ({final_auc} > {THRESHOLD}). Generating submission..."
        )

        # Compute final test predictions using the optimal weights
        final_test_preds = best_w * nb_test_preds + (1 - best_w) * neural_test_preds

        # Load test dataframe to preserve format
        test_df = load_data("test", load_cached_data=True)

        # Save submission
        save_submission(final_test_preds, test_df, output_dir="./submission")
        print("Submission saved to ./submission/submission.csv")
    else:
        print(
            f"\nPerformance check failed ({final_auc} <= {THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
