import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_tokenizer, get_train_dataloader, get_test_dataloader
from library.model import CustomXLMRoberta
from library.engine import get_optimizer, get_scheduler, train_fn, eval_fn, predict_fn
from library.inference import post_process_predictions


def create_mini_dataset(source_path, dest_path, num_samples=10):
    """
    Reads a source CSV and saves a small subset to dest_path for rapid testing.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)
    # Take a small subset
    mini_df = df.head(num_samples).copy()
    mini_df.to_csv(dest_path, index=False)
    print(f"Created mini dataset at {dest_path} with {len(mini_df)} samples.")
    return len(mini_df)


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    seed_everything(42)

    # Define temporary paths for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    # Override Config to use these mini files and run fast
    print("Overriding Config for speed...")
    Config.working_dir = demo_dir
    Config.output_dir = os.path.join(demo_dir, "output")
    Config.cache_dir = os.path.join(demo_dir, "cache")
    Config.submission_dir = os.path.join(demo_dir, "submission")

    # Create directories based on new config
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # Point to mini datasets
    Config.train_meta_path = mini_train_path
    Config.val_meta_path = mini_val_path
    Config.test_meta_path = mini_test_path
    Config.train_features_file = os.path.join(
        Config.cache_dir, "train_features.parquet"
    )
    Config.test_features_file = os.path.join(Config.cache_dir, "test_features.parquet")

    # Reduce computational load
    Config.epochs = 1
    Config.train_batch_size = 2
    Config.eval_batch_size = 2
    Config.use_full_train_data = False  # Only use train set, don't concat val
    Config.verbose = False

    # 2. Prepare Data
    # ---------------------------------------------------------
    print("\n[Step 1] Preparing Mini Datasets...")
    # We assume metadata files exist as per the problem description
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    create_mini_dataset(orig_train_path, mini_train_path, num_samples=10)
    create_mini_dataset(orig_val_path, mini_val_path, num_samples=5)
    create_mini_dataset(orig_test_path, mini_test_path, num_samples=5)

    # 3. Tokenizer & DataLoader
    # ---------------------------------------------------------
    print("\n[Step 2] Loading Tokenizer and DataLoaders...")
    tokenizer = get_tokenizer()

    # Force reload of cache by setting load_cached_data=False (or relying on new path)
    train_dataloader = get_train_dataloader(tokenizer, load_cached_data=False)

    # Verify DataLoader
    batch = next(iter(train_dataloader))
    print(f"Train Batch Keys: {list(batch.keys())}")

    # Assertions for Data
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert batch["input_ids"].shape[0] <= Config.train_batch_size
    assert batch["input_ids"].shape[1] == Config.max_length
    print("Data Loading verified successfully.")

    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing Model...")
    device = Config.device
    model = CustomXLMRoberta()
    model.to(device)

    # Verify Forward Pass
    print("Verifying Forward Pass...")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_positions = batch["start_positions"].to(device)
    end_positions = batch["end_positions"].to(device)
    relevance = batch["relevance"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        relevance=relevance,
    )

    # Assertions for Model
    assert "loss" in outputs, "Model output missing 'loss'"
    assert "start_logits" in outputs
    assert "end_logits" in outputs
    assert not torch.isnan(outputs["loss"]), "Loss is NaN"
    print(f"Forward pass successful. Loss: {outputs['loss'].item():.4f}")

    # 5. Training Loop (Engine)
    # ---------------------------------------------------------
    print("\n[Step 4] Testing Training Engine...")
    optimizer = get_optimizer(model)
    # Just a few steps for demo
    num_train_steps = len(train_dataloader) * Config.epochs
    scheduler = get_scheduler(optimizer, num_train_steps)

    # Run one epoch of training
    train_loss = train_fn(train_dataloader, model, optimizer, device, scheduler)
    print(f"Train Function executed. Average Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float)

    # Save model for inference test
    model_save_path = os.path.join(Config.output_dir, "model_seed_42.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[Step 5] Testing Inference Pipeline...")

    # Get test data
    test_dataloader, test_features_df = get_test_dataloader(
        tokenizer, load_cached_data=False
    )

    # Run prediction function
    start_logits, end_logits, rel_logits = predict_fn(test_dataloader, model, device)

    print(
        f"Logits Shapes: Start={start_logits.shape}, End={end_logits.shape}, Rel={rel_logits.shape}"
    )
    assert len(start_logits) == len(test_features_df)

    # Post-process
    print("Post-processing predictions...")
    predictions = post_process_predictions(
        test_features_df, start_logits, end_logits, rel_logits
    )

    # Verify Predictions
    print(f"Generated {len(predictions)} predictions.")
    sample_id = list(predictions.keys())[0]
    print(f"Sample Prediction -> ID: {sample_id}, Answer: '{predictions[sample_id]}'")

    assert isinstance(predictions, dict)
    assert len(predictions) > 0

    # 7. Cleanup
    # ---------------------------------------------------------
    print("\n[Step 6] Cleaning up...")
    # Optional: remove the demo directory to leave workspace clean
    # shutil.rmtree(demo_dir)
    print(f"Demo completed successfully. Artifacts stored in {demo_dir}")


if __name__ == "__main__":
    run_demo()
