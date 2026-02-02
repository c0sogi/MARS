import os
import sys
import pandas as pd
import numpy as np
import torch

# Import from the provided library files
from library.config import Config
from library.data_manager import get_data
from library.feature_extractor import SentenceEncoder, extract_embeddings
from library.model import SimilarityRegressor, generate_submission
from library.utils import compute_pearson_correlation


def main():
    print("Initializing configuration...")
    # 1. Setup and Speed Optimization
    Config.setup()

    # Enable DEBUG mode to use a small subset (500 samples) for fast demonstration
    Config.DEBUG = True
    print(f"Debug mode enabled: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("\nLoading data...")
    train_df = get_data("train", load_cached_data=True)
    val_df = get_data("val", load_cached_data=True)
    test_df = get_data("test", load_cached_data=True)

    # Verification: Check DataFrames
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    assert not train_df.empty, "Training dataframe is empty."
    assert "anchor_input" in train_df.columns, "Missing 'anchor_input' column."
    assert "target_input" in train_df.columns, "Missing 'target_input' column."
    assert "score" in train_df.columns, "Missing 'score' column in training data."

    # Verify preprocessing logic (Separator check)
    sample_input = train_df.iloc[0]["anchor_input"]
    assert (
        " [SEP] " in sample_input
    ), f"Separator missing in preprocessed input: {sample_input}"

    # 3. Feature Extraction
    print("\nInitializing Sentence Encoder...")
    encoder = SentenceEncoder()

    print("Extracting embeddings (this may take a moment)...")
    # Extract embeddings for all splits
    train_anchors, train_targets = extract_embeddings(train_df, encoder, "train")
    val_anchors, val_targets = extract_embeddings(val_df, encoder, "val")
    test_anchors, test_targets = extract_embeddings(test_df, encoder, "test")

    # Verification: Check Embedding Shapes
    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    expected_dim = 384
    print(f"Embedding shape (Train Anchors): {train_anchors.shape}")

    assert train_anchors.shape[0] == len(
        train_df
    ), "Mismatch between train rows and embedding count."
    assert (
        train_anchors.shape[1] == expected_dim
    ), f"Expected embedding dim {expected_dim}, got {train_anchors.shape[1]}"
    assert (
        train_targets.shape == train_anchors.shape
    ), "Anchor and Target embedding shapes do not match."

    # 4. Model Training
    print("\nTraining Similarity Regressor...")
    model = SimilarityRegressor()

    train_scores = train_df["score"].values
    val_scores = val_df["score"].values

    # Fit the model
    # This prints training and validation correlation internally
    model.fit(
        train_anchors,
        train_targets,
        train_scores,
        val_anchors=val_anchors,
        val_targets=val_targets,
        val_scores=val_scores,
    )

    # Verification: Check Prediction Logic
    # Test on a small batch to ensure outputs are clipped [0, 1]
    sample_preds = model.predict(val_anchors[:10], val_targets[:10])
    assert np.all(sample_preds >= 0.0) and np.all(
        sample_preds <= 1.0
    ), "Predictions out of range [0, 1]"
    print("Prediction logic verified (range [0, 1]).")

    # 5. Submission Generation
    print("\nGenerating submission...")
    generate_submission(model, test_df, test_anchors, test_targets)

    # Verification: Check Submission File
    submission_path = Config.SUBMISSION_FILE
    if os.path.exists(submission_path):
        print(f"Submission file created at: {submission_path}")
        sub_df = pd.read_csv(submission_path)

        # Check format
        assert list(sub_df.columns) == ["id", "score"], "Submission columns mismatch."
        assert len(sub_df) == len(
            test_df
        ), f"Submission length mismatch. Expected {len(test_df)}, got {len(sub_df)}"
        assert sub_df["score"].dtype in [
            float,
            np.float64,
            np.float32,
        ], "Score column is not float."

        print("Submission file format verified successfully.")
        print("\nTop 5 predictions:")
        print(sub_df.head())
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
