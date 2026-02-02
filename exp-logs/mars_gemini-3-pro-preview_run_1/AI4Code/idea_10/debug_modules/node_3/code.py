import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import kendall_tau_metric
from library.preprocess import FeatureExtractor
from library.dataset import CachedEmbeddingDataset, collate_fn
from library.train import Trainer
from library.inference import predict


def main():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Set a separate working directory for the demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.parquet")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Set hyperparams for speed
    Config.DEBUG_SAMPLE_SIZE = 20  # Process only 20 notebooks per split
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seed for reproducibility
    Config.set_seed(42)
    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Debug sample size: {Config.DEBUG_SAMPLE_SIZE}")

    # ------------------------------------------------------------------------
    # 2. Verify Metric Logic
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Kendall Tau Metric logic...")

    # Case 1: Perfect prediction
    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c d"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c d"]})
    score_perfect = kendall_tau_metric(df_pred_perfect, df_gt)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Case 2: Worst prediction (reversed)
    # n=4. Max swaps = n(n-1)/2 = 6. Formula denominator uses n(n-1) = 12.
    # Swaps needed for reversal = 6.
    # Score = 1 - 4 * (6 / 12) = 1 - 2 = -1.0
    df_pred_worst = pd.DataFrame({"id": ["nb1"], "cell_order": ["d c b a"]})
    score_worst = kendall_tau_metric(df_pred_worst, df_gt)
    assert np.isclose(score_worst, -1.0), f"Expected -1.0, got {score_worst}"

    print("Metric verification passed.")

    # ------------------------------------------------------------------------
    # 3. Feature Extraction
    # ------------------------------------------------------------------------
    print("\n[3] Running Feature Extraction (Preprocessing)...")
    extractor = FeatureExtractor()
    extractor.run()

    # Verify output files exist
    assert os.path.exists(Config.TRAIN_FEATURES_PATH), "Train features parquet missing"
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features parquet missing"
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test features parquet missing"

    # Verify content of one file
    df_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} train samples, found {len(df_train)}"
    assert "code_embeddings" in df_train.columns
    assert "markdown_embeddings" in df_train.columns
    print("Feature extraction successful.")

    # ------------------------------------------------------------------------
    # 4. Dataset and Collate Logic Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Dataset and Collate function...")
    train_ds = CachedEmbeddingDataset(Config.TRAIN_FEATURES_PATH, split_name="train")

    # Create a dummy batch
    batch_list = [train_ds[i] for i in range(min(3, len(train_ds)))]
    collated = collate_fn(batch_list)

    # Check keys
    required_keys = [
        "id",
        "code_embeddings",
        "markdown_embeddings",
        "labels",
        "code_mask",
        "markdown_mask",
        "code_lens",
        "markdown_lens",
    ]
    for key in required_keys:
        assert key in collated, f"Missing key in collated batch: {key}"

    # Verify Masking Logic
    # Mask should be False for valid tokens, True for padding
    code_lens = collated["code_lens"]
    code_mask = collated["code_mask"]

    for i, length in enumerate(code_lens):
        # Valid region check
        if length > 0:
            assert not code_mask[
                i, :length
            ].any(), f"Found masked values in valid region for sample {i}"
        # Padding region check
        if length < code_mask.size(1):
            assert code_mask[
                i, length:
            ].all(), f"Found unmasked values in padding region for sample {i}"

    print("Dataset and Collate logic verified.")

    # ------------------------------------------------------------------------
    # 5. Model Training
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop...")
    trainer = Trainer()
    trainer.run()

    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print("Training completed and model saved.")

    # ------------------------------------------------------------------------
    # 6. Inference
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference...")
    # Run prediction using the saved model
    predict(
        model_path=Config.MODEL_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        device=Config.DEVICE,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df_sub.columns
    assert "cell_order" in df_sub.columns
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if cell_order contains space-delimited strings
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order should be a string"
    assert len(sample_order.split()) > 0, "cell_order string is empty"

    print(f"Inference successful. Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
