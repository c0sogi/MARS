import os
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.data_factory import StreamBuilder
from library.model import DualStreamTrainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing Orthogonal-Physics Dual-Stream Pipeline...")
    set_seed(Config.SEED)

    # 2. Data Loading (Train & Validation)
    print("\n--- Loading Data ---")

    # Initialize Builders
    train_builder = StreamBuilder(mode="train")
    val_builder = StreamBuilder(mode="validation")

    # --- Stream A: Interaction ---
    print("Loading Stream A (Interaction) data...")
    X_train_A, y_train_A, ids_train_A = train_builder.build_interaction_set(
        load_cached=True
    )
    X_val_A, y_val_A, ids_val_A = val_builder.build_interaction_set(load_cached=True)

    # --- Stream B: Impact ---
    print("Loading Stream B (Impact) data...")
    X_train_B, y_train_B, ids_train_B = train_builder.build_impact_set(load_cached=True)
    X_val_B, y_val_B, ids_val_B = val_builder.build_impact_set(load_cached=True)

    # 3. Fast Baseline Constraint
    # Limit training data size to ensure execution finishes quickly (max 200k samples per stream)
    MAX_TRAIN_SAMPLES = 200000

    if len(X_train_A) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling Stream A Train from {len(X_train_A)} to {MAX_TRAIN_SAMPLES} for fast baseline..."
        )
        indices = np.random.choice(len(X_train_A), MAX_TRAIN_SAMPLES, replace=False)
        X_train_A = X_train_A.iloc[indices].reset_index(drop=True)
        y_train_A = y_train_A[indices]
        ids_train_A = ids_train_A[indices]

    if len(X_train_B) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling Stream B Train from {len(X_train_B)} to {MAX_TRAIN_SAMPLES} for fast baseline..."
        )
        indices = np.random.choice(len(X_train_B), MAX_TRAIN_SAMPLES, replace=False)
        X_train_B = X_train_B.iloc[indices].reset_index(drop=True)
        y_train_B = y_train_B[indices]
        ids_train_B = ids_train_B[indices]

    # 4. Training
    print("\n--- Training Models ---")
    trainer = DualStreamTrainer()
    trainer.fit(
        X_train_A, y_train_A, X_val_A, y_val_A, X_train_B, y_train_B, X_val_B, y_val_B
    )

    # Save checkpoint
    trainer.save_checkpoint()

    # 5. Validation Assessment
    print("\n--- Validation Assessment ---")

    # Predict Stream A
    prob_val_A = trainer.predict(X_val_A, stream_type="A")
    pred_val_A = (prob_val_A >= trainer.threshold_a).astype(int)

    # Predict Stream B
    prob_val_B = trainer.predict(X_val_B, stream_type="B")
    pred_val_B = (prob_val_B >= trainer.threshold_b).astype(int)

    # Combine for Global Metric
    # We concatenate the arrays. Since validation sets are distinct subsets of the full validation metadata
    # (Stream A = Player-Player, Stream B = Player-Ground), their union represents the full set.
    y_val_global = np.concatenate([y_val_A, y_val_B])
    pred_val_global = np.concatenate([pred_val_A, pred_val_B])

    final_metric = matthews_corrcoef(y_val_global, pred_val_global)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Analyze Stream A Errors
    print("Stream A (Interaction) Error Correlations:")
    error_A = np.abs(y_val_A - prob_val_A)
    # Select numeric columns only for correlation
    numeric_cols_A = X_val_A.select_dtypes(include=[np.number]).columns
    correlations_A = X_val_A[numeric_cols_A].corrwith(
        pd.Series(error_A, index=X_val_A.index)
    )
    print(correlations_A.abs().sort_values(ascending=False).head(5))

    # Analyze Stream B Errors
    print("\nStream B (Impact) Error Correlations:")
    error_B = np.abs(y_val_B - prob_val_B)
    numeric_cols_B = X_val_B.select_dtypes(include=[np.number]).columns
    correlations_B = X_val_B[numeric_cols_B].corrwith(
        pd.Series(error_B, index=X_val_B.index)
    )
    print(correlations_B.abs().sort_values(ascending=False).head(5))

    # 7. Submission
    THRESHOLD_SCORE = 0.6968
    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Load Test Data
        test_builder = StreamBuilder(mode="test")

        print("Loading Test Stream A...")
        X_test_A, _, ids_test_A = test_builder.build_interaction_set(load_cached=True)

        print("Loading Test Stream B...")
        X_test_B, _, ids_test_B = test_builder.build_impact_set(load_cached=True)

        # Inference
        print("Running Inference...")
        prob_test_A = trainer.predict(X_test_A, stream_type="A")
        pred_test_A = (prob_test_A >= trainer.threshold_a).astype(int)

        prob_test_B = trainer.predict(X_test_B, stream_type="B")
        pred_test_B = (prob_test_B >= trainer.threshold_b).astype(int)

        # Create DataFrames
        df_sub_A = pd.DataFrame({"contact_id": ids_test_A, "contact": pred_test_A})
        df_sub_B = pd.DataFrame({"contact_id": ids_test_B, "contact": pred_test_B})

        # Combine
        df_sub = pd.concat([df_sub_A, df_sub_B], ignore_index=True)

        # Ensure alignment with sample_submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge to enforce order and completeness
        # Left join on sample_submission ensures we have all required rows in correct order
        final_sub = sample_sub[["contact_id"]].merge(
            df_sub, on="contact_id", how="left"
        )

        # Fill missing (if any, though shouldn't be) with 0
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

        # Save
        final_sub.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
        print(f"Submission shape: {final_sub.shape}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD_SCORE}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
