import sys
import os
import pandas as pd
import numpy as np
import gc
from library.config import Config
from library.utils import setup_logging, calc_mcc
from library.feature_stream_a import StreamAFeatureGenerator
from library.feature_stream_b import StreamBFeatureGenerator
from library.model_factory import DualStreamModel


def main():
    # 1. Setup and Configuration
    # Modify Config for fast baseline execution as per requirements
    Config.TRAIN_CONFIG["num_boost_round"] = (
        2500  # Increased to prevent underfitting (Cite lesson 50)
    )
    Config.TRAIN_CONFIG["early_stopping_rounds"] = 50

    # Setup logging
    log_path = os.path.join(Config.WORKING_DIR, "run.log")
    logger = setup_logging(log_path)
    print("Starting Physically-Disentangled Dual-Stream Pipeline...")

    # 2. Feature Generation
    # Instantiate generators
    gen_a = StreamAFeatureGenerator()
    gen_b = StreamBFeatureGenerator()

    # Load/Generate Stream A (Interaction) Data
    print("\n--- Loading Stream A (Interaction) Data ---")
    train_a = gen_a.generate_features(mode="train", load_cached_data=True)
    val_a = gen_a.generate_features(mode="validation", load_cached_data=True)
    test_a = gen_a.generate_features(mode="test", load_cached_data=True)

    # Load/Generate Stream B (Impact) Data
    print("\n--- Loading Stream B (Impact) Data ---")
    train_b = gen_b.generate_features(mode="train", load_cached_data=True)
    val_b = gen_b.generate_features(mode="validation", load_cached_data=True)
    test_b = gen_b.generate_features(mode="test", load_cached_data=True)

    # 3. Data Preparation (Fast Baseline Constraints)
    # Limit training data size to ensure fast execution if datasets are huge
    # Increased to 600k to allow proper 10:1 undersampling (Cite lesson 60)
    MAX_TRAIN_SAMPLES = 600000

    if len(train_a) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling Stream A train data from {len(train_a)} to {MAX_TRAIN_SAMPLES} for fast baseline."
        )
        # Ensure we keep positives
        pos_a = train_a[train_a["contact"] == 1]
        neg_a = train_a[train_a["contact"] == 0].sample(
            n=MAX_TRAIN_SAMPLES - len(pos_a), random_state=Config.SEED
        )
        train_a = (
            pd.concat([pos_a, neg_a])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

    if len(train_b) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling Stream B train data from {len(train_b)} to {MAX_TRAIN_SAMPLES} for fast baseline."
        )
        pos_b = train_b[train_b["contact"] == 1]
        neg_b = train_b[train_b["contact"] == 0]
        # Stream B might have very few positives, ensure we don't crash if neg sample size is negative
        sample_size = max(0, MAX_TRAIN_SAMPLES - len(pos_b))
        sample_size = min(sample_size, len(neg_b))
        neg_b = neg_b.sample(n=sample_size, random_state=Config.SEED)
        train_b = (
            pd.concat([pos_b, neg_b])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

    # 4. Model Training
    print("\n--- Training Dual-Stream Model ---")
    model = DualStreamModel()
    model.train(train_a, train_b, val_a, val_b)

    # 5. Validation & Metric Calculation
    print("\n--- Validating ---")
    # Generate predictions on validation set
    # The predict method returns a dataframe with [contact_id, contact]
    preds_val = model.predict(val_a, val_b)

    # Reconstruct full ground truth from validation sets
    # Concatenate Stream A and Stream B validation sets
    val_full = pd.concat([val_a, val_b], axis=0)

    # Merge predictions with ground truth to ensure alignment
    # We use inner join to match predictions generated (in case some rows were dropped or filtered, though shouldn't be)
    val_merged = pd.merge(
        val_full[["contact_id", "contact"]],
        preds_val,
        on="contact_id",
        suffixes=("_true", "_pred"),
    )

    # Calculate MCC
    final_mcc = calc_mcc(val_merged["contact_true"], val_merged["contact_pred"])
    print(f"Final Validation Metric: {final_mcc:.16f}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Identify errors
    val_merged["error"] = (
        val_merged["contact_true"] != val_merged["contact_pred"]
    ).astype(int)

    # Merge errors back to feature dataframe to analyze correlations
    # We do this on the concatenated validation set containing features
    # Note: val_full has NaNs because Stream A and B have different columns.
    # Correlation ignores NaNs, which effectively isolates analysis per feature's relevant stream.
    analysis_df = pd.merge(
        val_full, val_merged[["contact_id", "error"]], on="contact_id"
    )

    # Select numeric features for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [
        "contact",
        "error",
        "step",
        "game_key",
        "play_id",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and "id" not in c]

    if feature_cols:
        correlations = (
            analysis_df[feature_cols + ["error"]].corr()["error"].drop("error")
        )
        print("Top 10 features correlated with error magnitude:")
        print(correlations.abs().sort_values(ascending=False).head(10))
    else:
        print("No numeric features available for correlation analysis.")

    # 7. Submission
    THRESHOLD = 0.6968
    if final_mcc > THRESHOLD:
        print(
            f"\nMetric ({final_mcc:.4f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        preds_test = model.predict(test_a, test_b)
        model.save_submission(preds_test)
    else:
        print(
            f"\nMetric ({final_mcc:.4f}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
