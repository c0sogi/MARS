import os
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functools import partial

# Suppress tqdm progress bars to keep output clean
from tqdm import tqdm

tqdm.__init__ = partial(tqdm.__init__, disable=True)

# Import library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data_loader import prepare_train_features, QADataset
from library.model import MuRILForQA
from library.trainer import QATrainer
from library.predictor import Predictor


def run_demonstration():
    print("=== Starting Q&A Pipeline Demonstration ===")

    # 1. Configuration Overrides for Speed
    # We create a separate working directory for the demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes to ensure fast execution
    Config.WORKING_DIR = DEMO_DIR
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_FOLDS = 1
    Config.DOC_STRIDE = 64  # Reduce stride for speed

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 2. Data Preparation Demo
    print("\n[Step 1] Loading and Processing Data...")

    # Load a tiny subset of training data (10 samples)
    if not os.path.exists(Config.META_TRAIN_PATH):
        raise FileNotFoundError(f"Metadata file not found: {Config.META_TRAIN_PATH}")

    df_train_full = pd.read_csv(Config.META_TRAIN_PATH)
    df_train_demo = df_train_full.head(10).copy()

    print(f"Loaded {len(df_train_demo)} training samples for demonstration.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # Process features
    # We use prepare_train_features directly to avoid caching overhead for this tiny set
    train_features = prepare_train_features(df_train_demo, tokenizer)

    # Verify features structure
    expected_cols = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "start_positions",
        "end_positions",
        "example_id",
    ]
    for col in expected_cols:
        assert (
            col in train_features.columns
        ), f"Missing column {col} in processed features"

    print(
        f"Generated {len(train_features)} features from 10 examples (due to sliding window)."
    )

    # Create Dataset and DataLoader
    train_dataset = QADataset(train_features, mode="train")
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False
    )

    # Verify DataLoader batch
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert batch["input_ids"].shape[1] == Config.MAX_LENGTH
    print("DataLoader verification successful.")

    # 3. Model Initialization Demo
    print("\n[Step 2] Initializing Model...")
    device = Config.DEVICE
    model = MuRILForQA()
    model.to(device)
    print(f"Model {Config.MODEL_NAME} initialized on {device}.")

    # 4. Training Loop Demo
    print("\n[Step 3] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Initialize Trainer
    trainer = QATrainer(model, tokenizer, device, optimizer=optimizer)

    # Run one training epoch
    train_loss = trainer.train_epoch(train_loader, epoch_idx=1)
    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run evaluation (using the same small set as validation for demo purposes)
    print("Running Evaluation...")
    # Note: eval_epoch expects raw_df to map example_id to answer_text for metric calculation
    val_jaccard, val_loss = trainer.eval_epoch(train_loader, df_train_demo)
    print(f"Validation Jaccard: {val_jaccard:.4f}")

    # Save the model to simulate a trained fold (required for Predictor)
    model_path = os.path.join(Config.WORKING_DIR, "model_fold_0.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model checkpoint saved to {model_path}")

    # 5. Inference/Prediction Demo
    print("\n[Step 4] Running Inference Pipeline...")

    # Create a dummy test file
    dummy_test_path = os.path.join(Config.WORKING_DIR, "dummy_test.csv")
    df_test_full = pd.read_csv(Config.META_TEST_PATH)
    df_test_demo = df_test_full.head(5).copy()
    df_test_demo.to_csv(dummy_test_path, index=False)

    # Override Config paths for inference so Predictor picks up our dummy file
    Config.TEST_PATH = dummy_test_path
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Initialize Predictor
    predictor = Predictor()

    # Run ensemble prediction (1 fold)
    # This will load the model we just saved, run inference on dummy_test.csv, and save submission.csv
    predictor.get_ensemble_predictions(folds=1)

    # 6. Verification
    print("\n[Step 5] Verifying Submission...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(submission_df.head())

    assert len(submission_df) == 5, f"Expected 5 predictions, got {len(submission_df)}"
    assert "id" in submission_df.columns and "PredictionString" in submission_df.columns

    # 7. Metric Utility Check
    print("\n[Step 6] Verifying Metric Function...")
    str1 = "This is a test answer"
    str2 = "This is a test"
    score = jaccard(str1, str2)
    # Intersection: {this, is, a, test} (4)
    # Union: {this, is, a, test, answer} (5)
    # Score: 4/5 = 0.8
    assert (
        abs(score - 0.8) < 1e-6
    ), f"Jaccard calculation incorrect. Expected 0.8, got {score}"
    print(f"Jaccard('{str1}', '{str2}') = {score}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
