import pandas as pd
import numpy as np
import sys
import os

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_manager import get_data
from library.model import CrossEncoderRegressor, generate_submission
from library.utils import compute_pearson_correlation


def main():
    # 1. Setup Environment
    Config.setup()

    # 2. Load Data
    # The data manager handles preprocessing (text_a, text_b) and caching
    train_df = get_data("train", load_cached_data=Config.LOAD_CACHED_DATA)
    val_df = get_data("val", load_cached_data=Config.LOAD_CACHED_DATA)
    test_df = get_data("test", load_cached_data=Config.LOAD_CACHED_DATA)

    # 3. Model Initialization and Training
    # Initialize Cross-Encoder
    model = CrossEncoderRegressor()

    # Train the model
    model.fit(train_df, val_df)

    # 4. Validation Assessment
    # Predict on validation set
    val_preds = model.predict(val_df)

    # Compute and print the final metric
    val_metric = compute_pearson_correlation(val_df["score"].values, val_preds)
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    val_df["pred"] = val_preds
    val_df["error"] = (val_df["score"] - val_df["pred"]).abs()

    # Generate simple meta-features for analysis
    val_df["anchor_len"] = val_df["anchor"].astype(str).str.len()
    val_df["target_len"] = val_df["target"].astype(str).str.len()
    val_df["len_diff"] = (val_df["anchor_len"] - val_df["target_len"]).abs()

    # Calculate correlation between error and features
    analysis_features = ["anchor_len", "target_len", "len_diff", "score"]
    correlations = val_df[analysis_features].corrwith(val_df["error"])

    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission Generation
    if val_metric > 0.8306522383964906:
        generate_submission(model, test_df)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print("Validation metric insufficient for submission.")


if __name__ == "__main__":
    main()
