import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb

# Import library components
from library.config import Config
from library.data_manager import DataBuilder
from library.model_factory import DualStreamModel
from library.evaluation import Evaluator
from library.utils import set_seed


def main():
    # 1. Setup and Configuration
    # Set global seed for reproducibility
    set_seed(Config.SEED)

    # Adjust configuration for a fast baseline execution
    # Reducing n_estimators to ensure training completes within time limits
    print("Configuring for fast baseline execution...")
    Config.XGB_PARAMS_A["n_estimators"] = 500
    Config.XGB_PARAMS_B["n_estimators"] = 500

    # 2. Data Loading
    print("Initializing Data Management...")
    data_builder = DataBuilder()

    # Load Training Data
    print("Loading Training Data...")
    train_data = data_builder.get_stream_data(split="train", load_cached_data=True)

    # Subsample training data to ensure strict runtime adherence (Fast Baseline)
    # Cap at 100,000 samples per stream
    MAX_TRAIN_SAMPLES = 100000
    for stream_key in ["stream_a", "stream_b"]:
        X_stream = train_data[stream_key]["X"]
        y_stream = train_data[stream_key]["y"]
        ids_stream = train_data[stream_key]["ids"]

        if len(y_stream) > MAX_TRAIN_SAMPLES:
            print(
                f"Subsampling {stream_key} from {len(y_stream)} to {MAX_TRAIN_SAMPLES} samples."
            )
            # Use seeded random choice
            indices = np.random.choice(len(y_stream), MAX_TRAIN_SAMPLES, replace=False)

            # Slice data
            train_data[stream_key]["X"] = X_stream.iloc[indices].reset_index(drop=True)
            train_data[stream_key]["y"] = y_stream[indices]
            train_data[stream_key]["ids"] = ids_stream[indices]

    # Load Validation Data (Full set required for valid metric)
    print("Loading Validation Data...")
    val_data = data_builder.get_stream_data(split="validation", load_cached_data=True)

    # 3. Model Training
    print("Initializing and Training Dual-Stream Model...")
    model = DualStreamModel()
    model.fit(train_data, val_data)

    # 4. Evaluation
    print("Performing Validation Inference...")
    # Generate predictions on the full validation set
    val_predictions = model.predict(val_data)

    # Construct Ground Truth for Validation
    # Stream A
    df_true_a = pd.DataFrame(
        {"contact_id": val_data["stream_a"]["ids"], "target": val_data["stream_a"]["y"]}
    )
    # Stream B
    df_true_b = pd.DataFrame(
        {"contact_id": val_data["stream_b"]["ids"], "target": val_data["stream_b"]["y"]}
    )

    # Combine and deduplicate (though IDs should be unique across streams)
    df_true = pd.concat([df_true_a, df_true_b], axis=0).drop_duplicates(
        subset=["contact_id"]
    )

    # Merge predictions with ground truth
    df_eval = df_true.merge(val_predictions, on="contact_id", how="inner")

    # Calculate MCC
    evaluator = Evaluator()
    final_mcc = evaluator.compute_mcc(
        df_eval["target"].values, df_eval["contact"].values
    )

    # Print required metric format
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    def analyze_stream_errors(stream_name, X, y_true, model_obj, threshold):
        print(f"Analyzing {stream_name}...")
        if len(y_true) == 0:
            print("No data for analysis.")
            return

        # Get probabilities
        probs = model._predict_proba(model_obj, X)
        preds = (probs >= threshold).astype(int)

        # Calculate Error Magnitude (0 for correct, 1 for incorrect)
        errors = np.abs(y_true - preds)

        # Calculate correlation between features and error
        # X is a DataFrame
        correlations = X.corrwith(pd.Series(errors, index=X.index))

        # Sort by absolute correlation
        top_corrs = correlations.abs().sort_values(ascending=False).head(5)

        print(f"Top 5 Features correlated with Error in {stream_name}:")
        print(top_corrs)

    # Analyze Stream A
    analyze_stream_errors(
        "Stream A (Player-Player)",
        val_data["stream_a"]["X"],
        val_data["stream_a"]["y"],
        model.model_a,
        model.threshold_a,
    )

    # Analyze Stream B
    analyze_stream_errors(
        "Stream B (Player-Ground)",
        val_data["stream_b"]["X"],
        val_data["stream_b"]["y"],
        model.model_b,
        model.threshold_b,
    )

    # 6. Submission
    SUBMISSION_THRESHOLD = 0.6565613438092561

    if final_mcc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation Metric ({final_mcc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        print("Loading Test Data...")
        test_data = data_builder.get_stream_data(split="test", load_cached_data=True)

        # Generate Predictions
        submission_df = model.predict(test_data)

        # Save
        save_path = Config.SUBMISSION_PATH
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation Metric ({final_mcc}) does not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
