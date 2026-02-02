import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_score
from library.train_stage1 import run_kfold
from library.meta_learner import prepare_stacking_data
from library.dataset import load_and_process_data


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Run Stage 1: Deep Semantic Learners (Ensemble)
    # We use epochs=2 to ensure execution finishes comfortably within the 2-hour limit
    # while still providing a strong baseline. debug=False ensures we use the full dataset.
    print(">>> Running Stage 1: DeBERTa-v3-Large Ensemble")
    run_kfold(debug=False, epochs=2)

    # 3. Stage 2: Meta-Learner Training
    print("\n>>> Running Stage 2: Stacking & Validation")

    # Load data prepared for stacking (OOF predictions + Structural Features + Context)
    # X_train: Matrix of features for the OOF set
    # y_train: Target scores for the OOF set
    # X_test: Matrix of features for the Test set
    # test_ids: IDs for the Test set
    X_train, y_train, X_test, test_ids = prepare_stacking_data(load_cached_data=True)

    # Train Ridge Regression Meta-Learner
    meta_model = Ridge(alpha=1.0, random_state=Config.SEED)
    meta_model.fit(X_train, y_train)

    # Generate calibrated predictions on the OOF set
    oof_preds_final = meta_model.predict(X_train)
    oof_preds_final = np.clip(oof_preds_final, 0.0, 1.0)

    # 4. Compute Final Validation Metric on Hold-out Validation Set
    # We need to map the OOF predictions back to the specific IDs in metadata/val.csv

    # Load original validation metadata to get the target IDs
    val_meta_df = pd.read_csv(Config.VAL_FILE)
    val_ids_set = set(val_meta_df["id"].values)

    # Reconstruct the ID order of X_train to filter correctly
    # We replicate the merge logic used in prepare_stacking_data
    oof_df = pd.read_csv(os.path.join(Config.WORKING_DIR, "stage1_oof.csv"))

    # Load processed metadata (cached)
    df_train_meta = load_and_process_data("train", load_cached_data=True)
    df_val_meta = load_and_process_data("val", load_cached_data=True)
    df_full_train = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Merge OOF with Metadata to align IDs with X_train rows
    train_merged = df_full_train.merge(oof_df[["id", "pred"]], on="id", how="inner")

    # Filter predictions for the validation set
    train_merged_ids = train_merged["id"].values
    val_mask = np.isin(train_merged_ids, list(val_ids_set))

    val_preds = oof_preds_final[val_mask]
    val_targets = y_train[val_mask]

    # Compute Metric
    final_metric = compute_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 5. Failure Analysis
    print("\n>>> Performing Failure Analysis on Validation Set")

    # Calculate residuals (Error Magnitude)
    residuals = np.abs(val_targets - val_preds)

    # Create analysis dataframe
    val_df_analysis = train_merged[val_mask].copy()
    val_df_analysis["error"] = residuals

    # Define features to check for correlation with error
    features_to_check = [
        "levenshtein_dist",
        "levenshtein_norm",
        "jaccard_sim",
        "len_diff",
        "word_len_diff",
    ]

    print("Correlation between Error Magnitude and Input Features:")
    for feat in features_to_check:
        if feat in val_df_analysis.columns:
            # Handle potential NaN or constant values safely
            try:
                if val_df_analysis[feat].std() > 0:
                    corr, _ = pearsonr(val_df_analysis["error"], val_df_analysis[feat])
                    print(f"  {feat}: {corr:.4f}")
                else:
                    print(f"  {feat}: NaN (Constant Feature)")
            except Exception:
                print(f"  {feat}: Error computing correlation")

    # 6. Submission Generation
    threshold = 0.8654320295612139

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold:.6f}). Generating Submission..."
        )

        # Predict on Test Set using the Meta-Learner
        test_preds = meta_model.predict(X_test)
        test_preds = np.clip(test_preds, 0.0, 1.0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "score": test_preds})

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({threshold:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
