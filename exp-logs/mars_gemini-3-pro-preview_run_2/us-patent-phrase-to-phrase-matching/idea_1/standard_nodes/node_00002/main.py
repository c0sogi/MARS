import pandas as pd
import numpy as np
import sys
import os

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_manager import get_data
from library.feature_extractor import SentenceEncoder, extract_embeddings
from library.model import SimilarityRegressor, generate_submission
from library.utils import compute_pearson_correlation


def main():
    # 1. Setup Environment
    Config.setup()

    # 2. Load Data
    # The data manager handles preprocessing (concatenating context) and caching
    train_df = get_data("train", load_cached_data=Config.LOAD_CACHED_DATA)
    val_df = get_data("val", load_cached_data=Config.LOAD_CACHED_DATA)
    test_df = get_data("test", load_cached_data=Config.LOAD_CACHED_DATA)

    # 3. Feature Extraction (Embeddings)
    # Initialize the frozen Bi-Encoder
    encoder = SentenceEncoder()

    # Extract embeddings for all splits
    # This uses the GPU if available and caches results to ./working/idea_1
    train_anchors, train_targets = extract_embeddings(
        train_df, encoder, "train", load_cached_data=Config.LOAD_CACHED_DATA
    )
    val_anchors, val_targets = extract_embeddings(
        val_df, encoder, "val", load_cached_data=Config.LOAD_CACHED_DATA
    )
    test_anchors, test_targets = extract_embeddings(
        test_df, encoder, "test", load_cached_data=Config.LOAD_CACHED_DATA
    )

    # 4. Model Training
    # Initialize Ridge Regression Head
    model = SimilarityRegressor()

    # Train the linear head on the frozen embeddings
    # We pass validation data here for internal logging, but we calculate the final metric explicitly below
    model.fit(
        train_anchors,
        train_targets,
        train_df["score"].values,
        val_anchors,
        val_targets,
        val_df["score"].values,
    )

    # 5. Validation Assessment
    # Predict on validation set
    val_preds = model.predict(val_anchors, val_targets)

    # Compute and print the final metric
    val_metric = compute_pearson_correlation(val_df["score"].values, val_preds)
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
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

    # 7. Submission Generation
    generate_submission(model, test_df, test_anchors, test_targets)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
