import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, jaccard
from library.model import get_model, get_tokenizer
from library.dataset import get_data, QADataset
from library.engine import train_one_epoch, validate_one_epoch, get_predictions
from library.inference import generate_submission


def run_pipeline_demo():
    print("=================================================================")
    print("       Question Answering Pipeline Demo (Debug Mode)             ")
    print("=================================================================")

    # -----------------------------------------------------------------------
    # 1. Configuration Setup
    # -----------------------------------------------------------------------
    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 30  # Small sample size for speed
    Config.EPOCHS = 1
    Config.N_FOLDS = 1  # Run only one fold
    Config.TRAIN_BATCH_SIZE = 4
    Config.EVAL_BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_CSV = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(
        f"Configuration: Device={device}, Debug={Config.DEBUG}, WorkDir={Config.WORKING_DIR}"
    )

    # -----------------------------------------------------------------------
    # 2. Metric Verification (utils.py)
    # -----------------------------------------------------------------------
    print("\n[1/6] Verifying Metric Logic...")
    str1 = "apple banana orange"
    str2 = "apple banana"
    # Intersection: {apple, banana} (2)
    # Union: {apple, banana, orange} (3)
    # Jaccard: 2/3 ~= 0.6667
    score = jaccard(str1, str2)
    expected_score = 2.0 / 3.0
    assert (
        abs(score - expected_score) < 1e-5
    ), f"Jaccard calculation failed: {score} != {expected_score}"
    print("      Jaccard metric test passed.")

    # -----------------------------------------------------------------------
    # 3. Data Processing (dataset.py)
    # -----------------------------------------------------------------------
    print("\n[2/6] Processing Data...")
    tokenizer = get_tokenizer()

    # Load Training Data (Debug subset)
    # load_cached_data=False forces processing from scratch to test logic
    train_df = get_data(tokenizer, split="train", load_cached_data=False)

    print(f"      Train features shape: {train_df.shape}")

    # Validate DataFrame structure
    required_cols = [
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "sequence_ids",
        "example_id",
    ]
    for col in required_cols:
        if col not in train_df.columns:
            raise AssertionError(f"Missing required column in processed data: {col}")

    if (
        "start_positions" not in train_df.columns
        or "end_positions" not in train_df.columns
    ):
        raise AssertionError("Training data missing label columns.")

    # Create Dataset and DataLoader
    train_dataset = QADataset(train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )

    # Verify a single batch
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] <= Config.TRAIN_BATCH_SIZE
    print("      Data loading and batching verified.")

    # -----------------------------------------------------------------------
    # 4. Model Initialization (model.py)
    # -----------------------------------------------------------------------
    print("\n[3/6] Initializing Model...")
    model = get_model()
    model.to(device)
    print("      Model loaded and moved to device.")

    # -----------------------------------------------------------------------
    # 5. Training Loop (engine.py)
    # -----------------------------------------------------------------------
    print("\n[4/6] Running Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Train
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch=0
    )
    print(f"      Training Loss: {train_loss:.4f}")

    if np.isnan(train_loss):
        raise ValueError("Training loss resulted in NaN.")

    # Save the model (required for inference step)
    # We save it as 'best_model_fold_0.pth' because Config.N_FOLDS=1 implies fold 0
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model_fold_0.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"      Model checkpoint saved to {model_save_path}")

    # -----------------------------------------------------------------------
    # 6. Validation Loop (engine.py)
    # -----------------------------------------------------------------------
    print("\n[5/6] Running Validation Loop...")

    # Load Validation Data
    val_df = get_data(tokenizer, split="val", load_cached_data=False)
    val_dataset = QADataset(val_df)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.EVAL_BATCH_SIZE, num_workers=0
    )

    # Validate
    val_loss = validate_one_epoch(model, val_loader, device)
    print(f"      Validation Loss: {val_loss:.4f}")

    # Test Prediction Generation
    start_logits, end_logits = get_predictions(model, val_loader, device)
    assert len(start_logits) == len(val_df), "Start logits length mismatch"
    assert len(end_logits) == len(val_df), "End logits length mismatch"
    print("      Prediction logits generation verified.")

    # -----------------------------------------------------------------------
    # 7. Inference and Submission (inference.py)
    # -----------------------------------------------------------------------
    print("\n[6/6] Generating Submission...")

    # Clear memory before inference
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # Run the full inference pipeline
    # This will load the saved model, process test data, and write submission.csv
    generate_submission()

    # Verify Output
    submission_path = Config.SUBMISSION_CSV
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"      Submission loaded. Shape: {sub_df.shape}")

    # Check format
    if list(sub_df.columns) != ["id", "PredictionString"]:
        raise AssertionError(f"Invalid submission columns: {sub_df.columns}")

    # Check content (PredictionString should be string)
    if not pd.api.types.is_string_dtype(sub_df["PredictionString"]):
        # It might be object if mixed, but let's check for nulls
        if sub_df["PredictionString"].isnull().any():
            # Fill NA with empty string just in case, though inference should handle it
            print(
                "      Warning: NaN found in PredictionString, this should be handled by inference code."
            )

    print(f"      Sample Prediction: {sub_df.iloc[0].to_dict()}")

    print("\n=================================================================")
    print("       Demo Completed Successfully")
    print("=================================================================")


if __name__ == "__main__":
    run_pipeline_demo()
