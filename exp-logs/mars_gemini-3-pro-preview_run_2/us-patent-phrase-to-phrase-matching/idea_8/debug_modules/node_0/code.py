import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_and_process_data, PhraseDataset
from library.model import DebertaV3WithFeatures
from library.train_stage1 import run_kfold
from library.meta_learner import train_and_predict_stacker


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> [1/6] Setting up configuration for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODELS_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Use a tiny model for rapid execution
    Config.MODEL_NAME = "prajjwal1/bert-tiny"

    # Reduce training parameters
    Config.EPOCHS = 1
    Config.FOLDS = 2
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUM_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid overhead for small data
    Config.DEBUG = True

    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Model: {Config.MODEL_NAME}")

    # =========================================================================
    # 2. Data Processing & Dataset Verification
    # =========================================================================
    print("\n>>> [2/6] Verifying Data Processing and Dataset...")

    # Load debug data (first 100 rows)
    # This tests load_and_process_data, generate_structural_features, and get_cpc_texts
    df_train = load_and_process_data("train", debug=True, debug_size=50)

    # Assertions for Data Processing
    assert not df_train.empty, "Processed dataframe is empty."
    expected_cols = [
        "id",
        "anchor",
        "target",
        "context",
        "score",
        "levenshtein_dist",
        "jaccard_sim",
        "context_text",
    ]
    for col in expected_cols:
        assert col in df_train.columns, f"Missing column: {col}"

    print(f"    Loaded {len(df_train)} training rows.")
    print("    Structural features computed successfully.")

    # Verify Dataset Class
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = PhraseDataset(df_train, tokenizer, max_length=32)

    # Fetch one item
    sample_item = dataset[0]

    # Assertions for Dataset
    assert "input_ids" in sample_item
    assert "structural_features" in sample_item
    assert "label" in sample_item
    assert sample_item["input_ids"].dim() == 1
    assert (
        sample_item["structural_features"].shape[0] == 6
    )  # 6 features defined in features.py

    print("    PhraseDataset verification passed.")

    # =========================================================================
    # 3. Model Architecture Verification
    # =========================================================================
    print("\n>>> [3/6] Verifying Model Architecture...")

    device = Config.DEVICE
    # Instantiate model with tiny backbone
    model = DebertaV3WithFeatures(
        num_features=6, num_classes=5, pretrained_model_name=Config.MODEL_NAME
    )
    model.to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    dummy_input_ids = (
        sample_item["input_ids"].unsqueeze(0).repeat(batch_size, 1).to(device)
    )
    dummy_mask = (
        sample_item["attention_mask"].unsqueeze(0).repeat(batch_size, 1).to(device)
    )
    dummy_feats = (
        sample_item["structural_features"].unsqueeze(0).repeat(batch_size, 1).to(device)
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input_ids, dummy_mask, dummy_feats)

    # Check output shape: [batch_size, num_classes]
    assert outputs.shape == (
        batch_size,
        5,
    ), f"Model output shape mismatch. Expected {(batch_size, 5)}, got {outputs.shape}"

    print("    Model forward pass successful.")
    del model, dummy_input_ids, dummy_mask, dummy_feats, outputs
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. Stage 1: K-Fold Training
    # =========================================================================
    print("\n>>> [4/6] Running Stage 1: K-Fold Training (Debug Mode)...")

    # This function orchestrates the training loop, saving models and OOF predictions
    run_kfold(debug=True, epochs=Config.EPOCHS)

    # Verify outputs of Stage 1
    oof_path = os.path.join(Config.WORKING_DIR, "stage1_oof.csv")
    test_pred_path = os.path.join(Config.WORKING_DIR, "stage1_test.csv")

    assert os.path.exists(oof_path), "Stage 1 OOF file was not created."
    assert os.path.exists(
        test_pred_path
    ), "Stage 1 Test predictions file was not created."

    print("    Stage 1 completed. OOF and Test predictions saved.")

    # =========================================================================
    # 5. Stage 2: Meta-Learner (Stacking)
    # =========================================================================
    print("\n>>> [5/6] Running Stage 2: Meta-Learner Stacking...")

    # This function loads Stage 1 outputs, trains a Ridge regressor, and creates submission
    train_and_predict_stacker(
        load_cached_data=False
    )  # Force recompute to use new debug data

    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), "Final submission file was not created."

    print("    Stage 2 completed. Submission file generated.")

    # =========================================================================
    # 6. Final Validation
    # =========================================================================
    print("\n>>> [6/6] Validating Final Submission...")

    submission = pd.read_csv(Config.SUBMISSION_FILE)

    # Check format
    assert list(submission.columns) == ["id", "score"], "Submission columns mismatch."

    # Check content
    # In debug mode, we used a subset of test data.
    # The run_kfold(debug=True) loads a subset of test.csv.
    # We verify that we have rows and scores are within bounds.
    assert len(submission) > 0, "Submission file is empty."

    scores = submission["score"]
    assert scores.min() >= 0.0, "Found scores below 0.0"
    assert scores.max() <= 1.0, "Found scores above 1.0"

    print(f"    Submission shape: {submission.shape}")
    print(
        f"    Score stats: Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}"
    )

    print("\n>>> DEMO COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_demo()
