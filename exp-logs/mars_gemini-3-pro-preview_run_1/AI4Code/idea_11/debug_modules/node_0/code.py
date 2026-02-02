import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config, set_seed
from library.utils import compute_kendall_tau, get_ranks
from library.data_preprocessing import Preprocessor
from library.dataset import CachedNotebookDataset
from library.model import DualContextAnchorNetwork
from library.train import Trainer
from library.inference import InferencePipeline


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Step 1: Configuring environment for demo run...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.SAMPLE_SIZE = 20  # Process only 20 notebooks
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup temporary working directory
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update dependent paths in Config to point to the demo directory
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print("Configuration updated successfully.")

    # ==========================================
    # 2. Unit Testing Helper Functions
    # ==========================================
    print("\nStep 2: Verifying utility functions...")

    # Test get_ranks
    # Scenario: 2 Code cells (ids: c1, c2) -> anchors at 0.0, 1.0
    # Markdown cells: m1 score 0.5 (between c1, c2), m2 score 1.5 (after c2)
    code_cells = ["c1", "c2"]
    pred_scores = {"m1": 0.5, "m2": 1.5}
    expected_order = "c1 m1 c2 m2"
    result_order = get_ranks(pred_scores, code_cells)
    assert (
        result_order == expected_order
    ), f"get_ranks failed. Got {result_order}, expected {expected_order}"

    # Test compute_kendall_tau
    # Perfect match
    df_gt = pd.DataFrame([{"id": "nb1", "cell_order": "a b c"}])
    df_pred_perfect = pd.DataFrame([{"id": "nb1", "cell_order": "a b c"}])
    score_perfect = compute_kendall_tau(df_pred_perfect, df_gt)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Kendall Tau perfect match failed. Got {score_perfect}"

    # Worst case (reversed) - 3 items: pairs=(a,b), (a,c), (b,c). 3 pairs.
    # GT: a b c. Pred: c b a.
    # Inversions in Pred relative to GT: (c,b), (c,a), (b,a) -> 3 inversions.
    # K = 1 - 4 * (3 / 6) = 1 - 2 = -1.0 ?? No, formula is 1 - 4 * S / (n(n-1)).
    # n=3, n(n-1)=6. S=3. K = 1 - 4*(3/6) = 1 - 2 = -1.
    df_pred_worst = pd.DataFrame([{"id": "nb1", "cell_order": "c b a"}])
    score_worst = compute_kendall_tau(df_pred_worst, df_gt)
    assert np.isclose(
        score_worst, -1.0
    ), f"Kendall Tau worst case failed. Got {score_worst}"

    print("Utility functions verified.")

    # ==========================================
    # 3. Preprocessing
    # ==========================================
    print("\nStep 3: Running Preprocessing (Feature Extraction)...")

    preprocessor = Preprocessor()
    # Run preprocessing for Train, Val, and Test
    # This will read metadata, load JSONs, encode text with MPNet, and save Parquet files
    preprocessor.run(load_cached_data=False)

    assert os.path.exists(
        Config.TRAIN_FEATURES_PATH
    ), "Train features file not created."
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features file not created."
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test features file not created."

    # Verify content of one file
    df_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
    assert not df_train.empty, "Train features dataframe is empty."
    assert "embedding" in df_train.columns, "Embeddings column missing."
    print(f"Preprocessing complete. Train features shape: {df_train.shape}")

    # ==========================================
    # 4. Dataset & DataLoader Verification
    # ==========================================
    print("\nStep 4: Verifying Dataset and DataLoader...")

    train_dataset = CachedNotebookDataset(Config.TRAIN_FEATURES_PATH, split="train")
    assert len(train_dataset) > 0, "Dataset is empty."

    # Check a single sample
    sample = train_dataset[0]
    required_keys = ["id", "code_embeddings", "md_embeddings", "labels"]
    for key in required_keys:
        assert key in sample, f"Sample missing key: {key}"

    # Check DataLoader collation
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=CachedNotebookDataset.collate_fn,
    )
    batch = next(iter(train_loader))

    assert "code_embeddings" in batch
    assert "code_mask" in batch
    assert "md_embeddings" in batch
    assert (
        batch["code_embeddings"].dim() == 3
    ), "Code embeddings should be (Batch, Seq, Dim)"
    assert batch["labels"].dim() == 2, "Labels should be (Batch, MD_Seq)"

    print("Dataset and DataLoader verified.")

    # ==========================================
    # 5. Model Verification
    # ==========================================
    print("\nStep 5: Verifying Model Architecture...")

    model = DualContextAnchorNetwork().to(Config.DEVICE)

    # Move batch to device
    code_emb = batch["code_embeddings"].to(Config.DEVICE)
    code_mask = batch["code_mask"].to(Config.DEVICE)
    code_lens = batch["code_lens"].to(Config.DEVICE)
    md_emb = batch["md_embeddings"].to(Config.DEVICE)
    md_mask = batch["md_mask"].to(Config.DEVICE)
    md_lens = batch["md_lens"].to(Config.DEVICE)

    # Forward pass
    logits = model(code_emb, code_mask, code_lens, md_emb, md_mask, md_lens)

    # Expected output: (Batch, Max_MD_Len, Max_Code_Len + 1)
    batch_size = code_emb.size(0)
    max_md_len = md_emb.size(1)
    max_code_len = code_emb.size(1)

    assert logits.shape == (
        batch_size,
        max_md_len,
        max_code_len + 1,
    ), f"Model output shape mismatch. Got {logits.shape}, expected {(batch_size, max_md_len, max_code_len + 1)}"

    print("Model forward pass successful.")

    # ==========================================
    # 6. Training Loop
    # ==========================================
    print("\nStep 6: Running Training Loop...")

    trainer = Trainer()
    # This will run for 1 epoch on the small subset and validate
    trainer.fit()

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Best model checkpoint was not saved."
    print("Training complete.")

    # ==========================================
    # 7. Inference
    # ==========================================
    print("\nStep 7: Running Inference...")

    inference_pipeline = InferencePipeline()
    # Run prediction on the test subset
    inference_pipeline.predict(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns mismatch."
    assert len(df_sub) > 0, "Submission file is empty."

    print(f"Inference complete. Submission generated at {Config.SUBMISSION_PATH}")
    print("\nAll demonstration steps passed successfully.")


if __name__ == "__main__":
    run_demo()
