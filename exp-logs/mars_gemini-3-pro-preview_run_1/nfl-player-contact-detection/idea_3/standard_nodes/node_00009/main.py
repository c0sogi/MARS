import pandas as pd
import numpy as np
import gc
import sys
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import seed_everything, print_metric
from library.data_pipeline import DataPipeline
from library.model_interaction import InteractionGBM
from library.model_impact import ImpactTrainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # Enforce reproducibility
    seed_everything(Config.SEED)

    # Adjust hyperparameters for a fast run (under 2 hours)
    # Reduce LightGBM rounds
    Config.LGBM_TRAIN_PARAMS["num_boost_round"] = 500
    Config.LGBM_TRAIN_PARAMS["early_stopping_rounds"] = 50

    # Reduce CNN epochs
    Config.CNN_TRAIN_PARAMS["epochs"] = 5

    print("Configuration initialized. Starting Split-Stream Pipeline...")

    # Initialize Data Pipeline
    pipeline = DataPipeline(Config)

    # -------------------------------------------------------------------------
    # 2. Stream A: Interaction Model (Player-Player)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STREAM A: INTERACTION MODEL TRAINING")
    print("=" * 40)

    # Load Stream A Data
    # Subsample training data to ensure quick execution while maintaining diversity
    train_df_a, feats_a = pipeline.get_stream_a_data("train", subsample_size=200000)
    val_df_a, _ = pipeline.get_stream_a_data("val")

    # Train LightGBM
    model_a = InteractionGBM(Config)
    model_a.train(train_df_a, val_df_a, feats_a)

    # Generate Validation Predictions for Stream A
    # (Used later for global metric and failure analysis)
    val_preds_a = model_a.predict(val_df_a, feats_a)
    val_df_a["pred_prob"] = val_preds_a

    # Cleanup to save memory
    del train_df_a
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. Stream B: Impact Model (Player-Ground)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STREAM B: IMPACT MODEL TRAINING")
    print("=" * 40)

    # Load Stream B Data
    # We use all ground contacts as they are naturally fewer than interactions
    X_train_b, y_train_b, _ = pipeline.get_stream_b_data("train")
    X_val_b, y_val_b, meta_val_b = pipeline.get_stream_b_data("val")

    # Train 1D-ResNet
    model_b = ImpactTrainer(Config)
    model_b.train(X_train_b, y_train_b, X_val_b, y_val_b)

    # Generate Validation Predictions for Stream B
    val_preds_b = model_b.predict(X_val_b)
    meta_val_b["pred_prob"] = val_preds_b

    # Cleanup
    del X_train_b, y_train_b
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Global Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("GLOBAL EVALUATION")
    print("=" * 40)

    # Combine predictions from both streams
    # val_df_a contains Player-Player rows
    # meta_val_b contains Player-Ground rows
    # We concat them to reconstruct the full validation set

    cols_to_keep = ["contact_id", "contact", "pred_prob"]

    df_eval_a = val_df_a[cols_to_keep].copy()
    df_eval_b = meta_val_b[cols_to_keep].copy()

    full_val = pd.concat([df_eval_a, df_eval_b], axis=0)

    y_true = full_val["contact"].values
    y_prob = full_val["pred_prob"].values
    y_pred = (y_prob > 0.5).astype(int)

    final_mcc = matthews_corrcoef(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate error magnitude
    full_val["error"] = np.abs(full_val["contact"] - full_val["pred_prob"])

    print("Global Error Stats:")
    print(full_val["error"].describe())

    # Stream A Feature Correlation Analysis
    # We analyze which features correlate most with errors in the Interaction model
    print("\nStream A (Interaction) Error Correlation with Features:")
    val_df_a["error"] = np.abs(val_df_a["contact"] - val_df_a["pred_prob"])

    # Compute correlation between features and error
    # feats_a contains the list of feature columns used
    corr_matrix = val_df_a[feats_a + ["error"]].corr()
    error_corrs = corr_matrix["error"].drop("error").sort_values(ascending=False)

    print("Top 5 Features associated with Error:")
    print(error_corrs.head(5))

    # -------------------------------------------------------------------------
    # 6. Submission Logic
    # -------------------------------------------------------------------------
    THRESHOLD = 0.5979349867102601

    if final_mcc > THRESHOLD:
        print("\n" + "=" * 40)
        print("GENERATING SUBMISSION")
        print("=" * 40)

        # 1. Stream A Inference (Test)
        print("Predicting Stream A (Test)...")
        test_df_a, _ = pipeline.get_stream_a_data("test")
        test_preds_a_prob = model_a.predict(test_df_a, feats_a)
        test_df_a["contact"] = (test_preds_a_prob > 0.5).astype(int)

        # 2. Stream B Inference (Test)
        print("Predicting Stream B (Test)...")
        X_test_b, _, meta_test_b = pipeline.get_stream_b_data("test")

        if len(meta_test_b) > 0:
            test_preds_b_prob = model_b.predict(X_test_b)
            meta_test_b["contact"] = (test_preds_b_prob > 0.5).astype(int)
        else:
            # Handle edge case if no ground contacts in test (unlikely)
            meta_test_b = pd.DataFrame(columns=["contact_id", "contact"])

        # 3. Merge and Format
        sub_a = test_df_a[["contact_id", "contact"]]
        sub_b = meta_test_b[["contact_id", "contact"]]

        submission_preds = pd.concat([sub_a, sub_b], axis=0)

        # Load sample submission to ensure correct order and completeness
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into sample submission structure
        # Left join ensures we keep all rows from sample_sub
        final_submission = sample_sub[["contact_id"]].merge(
            submission_preds, on="contact_id", how="left"
        )

        # Fill missing (if any) with 0 and ensure int type
        final_submission["contact"] = final_submission["contact"].fillna(0).astype(int)

        # Save
        final_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {final_submission.shape}")

    else:
        print(
            f"\nValidation Metric ({final_mcc}) did not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
