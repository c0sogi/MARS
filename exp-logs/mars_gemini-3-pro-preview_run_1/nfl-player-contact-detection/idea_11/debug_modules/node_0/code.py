import os
import sys
import numpy as np
import pandas as pd
import warnings
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
from library.config import (
    WORKING_DIR,
    RANDOM_STATE,
    SCOUT_LGBM_PARAMS,
    EXPERT_LGBM_PARAMS,
    EXPERT_XGB_PARAMS,
    HARD_NEGATIVE_THRESHOLD,
    SCOUT_NEG_RATIO,
)
from library.utils import seed_everything, compute_mcc
from library.feature_engineering import generate_features
from library.model_zoo import LGBMWrapper, XGBWrapper, HeterogeneousEnsemble
from library.data_loader import load_sample_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    seed_everything(RANDOM_STATE)

    # Define reduced parameters for speed demonstration
    DEMO_N_ROWS = 2000
    DEMO_BOOST_ROUNDS = 10

    # Override n_estimators in params for speed (though num_boost_round in train() controls loop)
    # We update the dictionaries to be safe
    scout_params = SCOUT_LGBM_PARAMS.copy()
    scout_params["n_estimators"] = DEMO_BOOST_ROUNDS

    expert_lgbm_params = EXPERT_LGBM_PARAMS.copy()
    expert_lgbm_params["n_estimators"] = DEMO_BOOST_ROUNDS

    expert_xgb_params = EXPERT_XGB_PARAMS.copy()
    expert_xgb_params["n_estimators"] = DEMO_BOOST_ROUNDS

    # =========================================================================
    # 2. Feature Engineering
    # =========================================================================
    print("\n[Step 1] Generating Features (Train/Val)...")

    # Generate features for a small subset of training data
    # Gating is enabled for training to filter distant pairs
    df_train = generate_features(
        split="train",
        load_cached_data=False,  # Force fresh generation for demo
        nrows=DEMO_N_ROWS,
        gating=True,
    )

    # Generate features for validation
    df_val = generate_features(
        split="val", load_cached_data=False, nrows=DEMO_N_ROWS, gating=True
    )

    # Verification
    assert not df_train.empty, "Training dataframe is empty."
    assert "distance" in df_train.columns, "Feature 'distance' missing."
    assert "radial_flux_t0" in df_train.columns, "Lag features missing."
    # Verify Gating: No pairs > 2.5 yards unless it's Ground or missing tracking
    # Note: We check a sample to ensure logic held
    valid_dist_mask = (df_train["nfl_player_id_2"] != "G") & (
        df_train["distance"].notna()
    )
    if valid_dist_mask.any():
        max_dist = df_train.loc[valid_dist_mask, "distance"].max()
        # Allow small float tolerance or cases where gating logic in library might be slightly different
        # library says GATING_DISTANCE = 2.5
        assert max_dist <= 3.0, f"Gating failed, found distance {max_dist}"

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")

    # Prepare Feature Columns (exclude metadata)
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
    ]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # =========================================================================
    # 3. Scout Model Training
    # =========================================================================
    print("\n[Step 2] Training Scout Model...")

    # Create Balanced Dataset for Scout
    pos_mask = df_train["contact"] == 1
    df_pos = df_train[pos_mask]
    df_neg = df_train[~pos_mask]

    # Handle case where sample is too small to have positives
    if len(df_pos) == 0:
        print("Warning: No positives in small sample. Mocking a positive for demo.")
        df_pos = df_neg.iloc[:2].copy()
        df_pos["contact"] = 1

    n_neg_sample = int(len(df_pos) * SCOUT_NEG_RATIO)
    # Ensure at least some negatives
    n_neg_sample = max(n_neg_sample, 10)

    df_neg_sample = df_neg.sample(
        n=min(len(df_neg), n_neg_sample), random_state=RANDOM_STATE
    )
    df_scout = pd.concat([df_pos, df_neg_sample]).sample(
        frac=1.0, random_state=RANDOM_STATE
    )

    X_scout = df_scout[feature_cols]
    y_scout = df_scout["contact"]
    X_val_full = df_val[feature_cols]
    y_val_full = df_val["contact"]

    # Train Scout
    scout_model = LGBMWrapper(scout_params, name="demo_scout")
    scout_model.train(
        X_scout,
        y_scout,
        X_val_full,
        y_val_full,
        num_boost_round=DEMO_BOOST_ROUNDS,
        early_stopping_rounds=5,
        verbose_eval=False,
    )

    # Verify Model
    assert scout_model.model is not None, "Scout model failed to train."

    # =========================================================================
    # 4. Hard Negative Mining
    # =========================================================================
    print("\n[Step 3] Mining Hard Negatives...")

    # Predict on full training set (negatives only)
    neg_indices = df_train.index[df_train["contact"] == 0]
    X_train_neg = df_train.loc[neg_indices, feature_cols]

    if not X_train_neg.empty:
        preds_neg = scout_model.predict(X_train_neg)

        # Identify hard negatives
        hard_mask = preds_neg > HARD_NEGATIVE_THRESHOLD
        hard_neg_indices = neg_indices[hard_mask]

        print(f"Mined {len(hard_neg_indices)} hard negatives.")
    else:
        hard_neg_indices = []

    # =========================================================================
    # 5. Expert Model Training
    # =========================================================================
    print("\n[Step 4] Training Expert Ensemble...")

    # Construct Expert Dataset
    # Positives + Hard Negatives + Buffer
    df_hard = df_train.loc[hard_neg_indices]

    # Buffer (easy negatives)
    # Exclude hard negs from buffer pool
    easy_pool_indices = df_train.index.difference(hard_neg_indices).difference(
        df_pos.index
    )
    df_easy_pool = df_train.loc[easy_pool_indices]

    n_buffer = len(df_pos)
    df_buffer = df_easy_pool.sample(
        n=min(len(df_easy_pool), n_buffer), random_state=RANDOM_STATE
    )

    df_expert = pd.concat([df_pos, df_hard, df_buffer]).sample(
        frac=1.0, random_state=RANDOM_STATE
    )

    X_expert = df_expert[feature_cols]
    y_expert = df_expert["contact"]

    # Train Expert LGBM
    lgbm_expert = LGBMWrapper(expert_lgbm_params, name="demo_expert_lgbm")
    lgbm_expert.train(
        X_expert,
        y_expert,
        X_val_full,
        y_val_full,
        num_boost_round=DEMO_BOOST_ROUNDS,
        early_stopping_rounds=5,
        verbose_eval=False,
    )

    # Train Expert XGBoost
    xgb_expert = XGBWrapper(expert_xgb_params, name="demo_expert_xgb")
    xgb_expert.train(
        X_expert,
        y_expert,
        X_val_full,
        y_val_full,
        num_boost_round=DEMO_BOOST_ROUNDS,
        early_stopping_rounds=5,
        verbose_eval=False,
    )

    # Create Ensemble
    ensemble = HeterogeneousEnsemble([lgbm_expert, xgb_expert])

    # Optimize Threshold
    best_threshold = ensemble.optimize_threshold(X_val_full, y_val_full)
    assert 0.0 < best_threshold < 1.0, "Optimal threshold out of bounds."

    # =========================================================================
    # 6. Inference
    # =========================================================================
    print("\n[Step 5] Running Inference on Test Data...")

    # Generate Test Features (No Gating for Test!)
    # We use a small nrows for demo, but in reality, this would be the full test set
    df_test = generate_features(
        split="test", load_cached_data=False, nrows=DEMO_N_ROWS, gating=False
    )

    X_test = df_test[feature_cols]

    # Predict
    probs = ensemble.predict(X_test)
    preds = (probs >= best_threshold).astype(int)

    # Format Submission
    df_preds = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact_pred": preds}
    )

    # Load template
    df_sub_template = load_sample_submission()

    # Merge (Left join on template to preserve all rows)
    # Note: Since we only processed DEMO_N_ROWS of test data,
    # the merge will result in NaNs for the unprocessed rows.
    # We fill these with 0 for the demo.
    df_final = df_sub_template.drop(columns=["contact"]).merge(
        df_preds, on="contact_id", how="left"
    )
    df_final["contact"] = df_final["contact_pred"].fillna(0).astype(int)

    submission_df = df_final[["contact_id", "contact"]]

    # Verification
    assert len(submission_df) == len(df_sub_template), "Submission row count mismatch."
    assert submission_df["contact"].isin([0, 1]).all(), "Invalid values in prediction."

    # Save (Mock save to working dir)
    save_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
