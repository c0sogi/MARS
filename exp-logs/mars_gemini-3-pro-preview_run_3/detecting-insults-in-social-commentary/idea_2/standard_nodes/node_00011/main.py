import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add the current directory to path to ensure library imports work if not implicitly handled
sys.path.append(os.getcwd())

from library.utils import set_seed, load_data, save_submission
from library.execution import run_neural
from sklearn.metrics import roc_auc_score


def main():
    # 1. Setup
    set_seed(42)
    warnings.filterwarnings("ignore")

    # 2. Run Neural Stream (RoBERTa-Large)
    # Cite solution_lesson_node_00007: Vertical scaling (Large model) + Mean Pooling
    # Cite solution_lesson_node_00006: Removing statistical stream as ensemble gain is negligible
    print("Executing Neural Stream (RoBERTa-Large)...")
    neural_val_preds, neural_test_preds = run_neural(
        load_cached_data=True,
        max_samples=None,
        epochs=3,
        batch_size=16,
        lr=1e-5,  # Lower learning rate for larger model
        model_name="roberta-large",
    )

    # 3. Load Validation Labels
    val_df = load_data("val", load_cached_data=True)
    y_val = val_df["Insult"].values

    # 4. Calculate Validation Metric
    final_auc = roc_auc_score(y_val, neural_val_preds)

    # 5. Report Validation Metric (Strict Format)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude
    errors = np.abs(y_val - neural_val_preds)

    # Feature 1: Character Length
    # Ensure comment is string to avoid len() errors on potential NaNs (though load_data cleans this)
    val_df["char_len"] = val_df["Comment"].astype(str).apply(len)
    corr_char = np.corrcoef(val_df["char_len"], errors)[0, 1]
    print(f"Correlation between Error and Character Length: {corr_char}")

    # Feature 2: Word Count
    val_df["word_count"] = val_df["Comment"].astype(str).apply(lambda x: len(x.split()))
    corr_word = np.corrcoef(val_df["word_count"], errors)[0, 1]
    print(f"Correlation between Error and Word Count: {corr_word}")

    # 7. Conditional Submission
    THRESHOLD = 0.9639408866995074

    if final_auc > THRESHOLD:
        print(
            f"\nPerformance check passed ({final_auc} > {THRESHOLD}). Generating submission..."
        )

        # Compute final test predictions
        final_test_preds = neural_test_preds

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
