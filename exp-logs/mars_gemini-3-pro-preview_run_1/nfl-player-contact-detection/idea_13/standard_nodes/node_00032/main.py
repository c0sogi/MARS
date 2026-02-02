import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from library.config import SEED, SUBMISSION_PATH
from library.utils import seed_everything, compute_mcc
from library.data_loader import get_data
from library.feature_engine import generate_features
from library.mining_curriculum import MiningCurriculum
from library.model_factory import UnifiedEnsemble


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Starting IKS-ME Pipeline...")

    # 2. Load Gated Training Data
    # We use gating (The Sieve) to filter trivial non-contacts for training efficiency
    print("\n--- Phase 1: Data Loading (Gated) ---")
    df_train = get_data("train", apply_gating=True)
    df_val_gated = get_data("val", apply_gating=True)

    # 3. Feature Engineering
    print("\n--- Phase 2: Feature Engineering ---")
    df_train = generate_features(df_train, "train")
    df_val_gated = generate_features(df_val_gated, "val")

    # 4. Define Feature Columns
    # Exclude metadata, IDs, and target
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "datetime",
        "contact",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
        "p2_join",
        "row_id",
        "fold",
        "nfl_player_id",
        "p1_join_id",
        "p2_join_id",
    ]
    # Select numeric columns that are not in exclude list
    feature_cols = [
        c
        for c in df_train.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])
    ]

    print(f"Identified {len(feature_cols)} features for training.")

    # 5. Mining Curriculum
    print("\n--- Phase 3: Mining Curriculum ---")
    mc = MiningCurriculum(feature_cols, target_col="contact")

    # A. Run Scout Mining to find Hard Negatives
    # This trains a scout model and predicts on the full gated train set
    hard_neg_indices = mc.run_scout_mining(df_train, df_val_gated)

    # B. Construct Expert Dataset
    # Combines Positives + Hard Negatives + Random Buffer
    df_expert = mc.prepare_expert_dataset(df_train, hard_neg_indices)

    # Free up memory from full df_train if possible, though we might need it?
    # Actually df_expert is a copy, so we can delete df_train to save RAM for the ensemble
    del df_train
    gc.collect()

    # 6. Train Expert Ensemble
    print("\n--- Phase 4: Expert Model Training ---")
    model = UnifiedEnsemble()
    model.fit(df_expert, df_val_gated, feature_cols)

    # Cleanup Expert Data
    del df_expert, df_val_gated
    gc.collect()

    # 7. Full Validation & Threshold Optimization
    print("\n--- Phase 5: Full Validation & Evaluation ---")
    # We must evaluate on the FULL validation set (ungated) for a correct metric
    df_val_full = get_data("val", apply_gating=False)
    df_val_full = generate_features(df_val_full, "val")

    print(f"Predicting on full validation set ({len(df_val_full)} rows)...")
    val_preds = model.predict(df_val_full, feature_cols)
    y_val = df_val_full["contact"].values

    # Optimize Threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    best_mcc = -1.0
    best_thresh = 0.5

    for t in thresholds:
        mcc = compute_mcc(y_val, (val_preds > t).astype(int))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    print(f"Final Validation Metric: {best_mcc}")
    print(f"Best Threshold: {best_thresh}")

    # 8. Failure Analysis
    print("\n--- Phase 6: Failure Analysis ---")
    # Calculate error magnitude
    df_val_full["error"] = np.abs(y_val - val_preds)

    # Compute correlations
    corrs = {}
    for feat in feature_cols:
        if feat in df_val_full.columns:
            # Drop NaNs for correlation calculation
            valid_mask = df_val_full[feat].notna() & df_val_full["error"].notna()
            if valid_mask.sum() > 0:
                corrs[feat] = np.corrcoef(
                    df_val_full.loc[valid_mask, feat],
                    df_val_full.loc[valid_mask, "error"],
                )[0, 1]
            else:
                corrs[feat] = 0.0

    # Sort by correlation strength (absolute or positive? usually positive correlation with error means feature causes error)
    # We look for high positive correlation with error
    sorted_corrs = sorted(corrs.items(), key=lambda x: x[1], reverse=True)

    print("Top 5 Features associated with high prediction error:")
    for name, val in sorted_corrs[:5]:
        print(f"  {name}: {val:.4f}")

    # Cleanup Validation Data
    del df_val_full
    gc.collect()

    # 9. Submission
    print("\n--- Phase 7: Submission ---")
    TARGET_METRIC = 0.6782

    if best_mcc > TARGET_METRIC:
        print(
            f"Validation MCC ({best_mcc:.4f}) > {TARGET_METRIC}. Generating submission..."
        )

        # Load Test Data (No Gating)
        df_test = get_data("test", apply_gating=False)
        df_test = generate_features(df_test, "test")

        # Predict
        test_probs = model.predict(df_test, feature_cols)
        test_preds = (test_probs > best_thresh).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": test_preds}
        )

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH} with {len(submission)} rows.")

    else:
        print(
            f"Validation MCC ({best_mcc:.4f}) did not meet threshold ({TARGET_METRIC}). Skipping submission generation."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
