import sys
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import setup_seed, compute_mcc
from library.training import Trainer
from library.inference import Predictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing Run...")
    setup_seed(Config.SEED)

    # 2. Training
    # Initialize Trainer. Config.DEBUG determines if we use a subset of data.
    # The Trainer handles data loading, feature engineering, and model training.
    trainer = Trainer(debug=Config.DEBUG)

    print("Starting Training Pipeline...")
    # results contains models, thresholds, scores, and feature lists for streams 'A' and 'B'
    results = trainer.train()

    if not results:
        print("Training failed to produce results.")
        sys.exit(1)

    # 3. Global Validation & Failure Analysis
    print("\n=== Performing Global Validation & Failure Analysis ===")

    # We need to reconstruct the full validation set to calculate the global MCC
    # and perform failure analysis. We leverage the trainer's data loader and prep methods.

    # Load raw data needed for feature generation
    df_val_meta = trainer.dl.load_metadata("validation")
    df_tracking = trainer.dl.load_tracking(
        "train"
    )  # Validation tracking is in train file
    df_helmets = trainer.dl.load_helmets(
        "train"
    )  # Validation helmets are in train file

    val_preds = []
    val_labels = []

    # Store error analysis data
    error_correlations = {}

    # --- Process Stream A (Collider) ---
    if "A" in results:
        print("Evaluating Stream A (Player-Player)...")
        model_A = results["A"]["model"]
        thresh_A = results["A"]["threshold"]
        feats_A = results["A"]["features"]

        # Prepare validation data (no undersampling for validation)
        df_val_A = trainer._prepare_stream_data(
            "A", df_val_meta, df_tracking, df_helmets, is_train=False
        )

        if not df_val_A.empty:
            X_val_A = df_val_A[feats_A]
            y_val_A = df_val_A["contact"].values

            # Predict
            probs_A = model_A.predict_proba(X_val_A)[:, 1]
            preds_A = (probs_A >= thresh_A).astype(int)

            val_preds.append(preds_A)
            val_labels.append(y_val_A)

            # Failure Analysis Stream A
            # Calculate error (0 for correct, 1 for incorrect)
            errors_A = np.abs(y_val_A - preds_A)
            # Calculate correlation between features and error
            # We use a temporary dataframe for correlation computation
            df_analysis_A = X_val_A.copy()
            df_analysis_A["__error__"] = errors_A
            corr_A = df_analysis_A.corrwith(df_analysis_A["__error__"]).drop(
                "__error__"
            )
            error_correlations["A"] = corr_A.sort_values(ascending=False).head(5)

    # --- Process Stream B (Accelerometer) ---
    if "B" in results:
        print("Evaluating Stream B (Player-Ground)...")
        model_B = results["B"]["model"]
        thresh_B = results["B"]["threshold"]
        feats_B = results["B"]["features"]

        # Prepare validation data
        df_val_B = trainer._prepare_stream_data(
            "B", df_val_meta, df_tracking, df_helmets, is_train=False
        )

        if not df_val_B.empty:
            X_val_B = df_val_B[feats_B]
            y_val_B = df_val_B["contact"].values

            # Predict
            probs_B = model_B.predict_proba(X_val_B)[:, 1]
            preds_B = (probs_B >= thresh_B).astype(int)

            val_preds.append(preds_B)
            val_labels.append(y_val_B)

            # Failure Analysis Stream B
            errors_B = np.abs(y_val_B - preds_B)
            df_analysis_B = X_val_B.copy()
            df_analysis_B["__error__"] = errors_B
            corr_B = df_analysis_B.corrwith(df_analysis_B["__error__"]).drop(
                "__error__"
            )
            error_correlations["B"] = corr_B.sort_values(ascending=False).head(5)

    # --- Compute Global Metrics ---
    if val_preds:
        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_labels)

        global_mcc = compute_mcc(y_true_all, y_pred_all)

        # REQUIRED OUTPUT FORMAT
        print(f"Final Validation Metric: {global_mcc}")

        # Print Failure Analysis
        print("\n--- Failure Analysis (Top Features Correlated with Error) ---")
        for stream, corrs in error_correlations.items():
            print(f"Stream {stream}:")
            print(corrs.to_string())
            print("-" * 20)

    else:
        print("Error: No validation predictions generated.")
        global_mcc = 0.0

    # 4. Submission
    # Threshold check as per requirements
    SUBMISSION_THRESHOLD = 0.6968

    if global_mcc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({global_mcc}) > {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        # Initialize Predictor with trained models and optimized thresholds
        predictor = Predictor(
            model_a=results.get("A", {}).get("model"),
            thresh_a=results.get("A", {}).get("threshold", 0.5),
            model_b=results.get("B", {}).get("model"),
            thresh_b=results.get("B", {}).get("threshold", 0.5),
            debug=Config.DEBUG,
        )

        # Run inference pipeline
        predictor.predict()

    else:
        print(
            f"\nValidation metric ({global_mcc}) <= {SUBMISSION_THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
