import os
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import provided library modules
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import EssayRegressor
from library.engine import train_model, generate_submission
from library.utils import compute_qwk


def run_demo():
    # 1. Setup and Configuration
    print("=== Setting up Environment ===")
    seed_everything(42)
    transformers.logging.set_verbosity_error()

    # Override Config for a fast demonstration
    Config.debug = True  # Limits data to 50 samples per split
    Config.epochs = 1  # Run only 1 epoch
    Config.train_batch_size = 4
    Config.valid_batch_size = 8

    # Update paths to use a specific demo directory in working/
    demo_dir = "./working/demo"
    os.makedirs(demo_dir, exist_ok=True)
    Config.working_dir = demo_dir
    Config.model_save_path = os.path.join(demo_dir, "model_demo.pth")
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    # Note: Config.train_cache_path etc. are static class attributes.
    # We will disable cache loading in get_dataloaders to avoid path issues.

    print(f"Device: {Config.device}")
    print("Configuration updated for speed (Debug=True, Epochs=1).")

    # 2. Data Preparation
    print("\n=== Preparing Data ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load dataloaders (disable cache to force fresh load from metadata CSVs)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    # Verify DataLoaders
    try:
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] <= Config.train_batch_size
        print("DataLoaders created and verified successfully.")
    except Exception as e:
        print(f"DataLoader verification failed: {e}")
        raise

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = EssayRegressor(model_name=Config.model_name, pretrained=True)
    model.to(Config.device)

    # Verify Model Forward Pass
    try:
        with torch.no_grad():
            dummy_ids = batch["input_ids"].to(Config.device)
            dummy_mask = batch["attention_mask"].to(Config.device)
            output = model(dummy_ids, dummy_mask)
            # Output shape should be (batch_size, 1)
            assert output.shape == (dummy_ids.size(0), 1)
        print("Model forward pass verified successfully.")
    except Exception as e:
        print(f"Model verification failed: {e}")
        raise

    # 4. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * Config.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    print("\n=== Starting Training ===")
    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        epochs=Config.epochs,
        patience=1,
    )
    print("Training loop finished.")

    # 6. Metric Verification
    print("\n=== Verifying Metric (QWK) ===")
    # Test case: Perfect correlation should yield 1.0
    # Predictions: 3.1 -> 3, 4.9 -> 5. Matches ground truth.
    y_true_test = [3, 5, 1, 2]
    y_pred_test = [3.1, 4.9, 1.2, 1.8]
    score = compute_qwk(y_true_test, y_pred_test)
    assert score > 0.95, f"QWK calculation incorrect. Expected ~1.0, got {score}"
    print(f"Metric verification passed. Calculated QWK: {score}")

    # 7. Inference and Submission
    print("\n=== Generating Submission ===")
    generate_submission(
        trained_model, test_loader, Config.device, output_path=Config.submission_path
    )

    # Verify Submission File
    if os.path.exists(Config.submission_path):
        df_sub = pd.read_csv(Config.submission_path)
        print(f"Submission file created at {Config.submission_path}")
        print(f"Rows: {len(df_sub)}")
        print(df_sub.head())

        # Assertions
        assert list(df_sub.columns) == ["essay_id", "score"]
        assert len(df_sub) > 0
        assert df_sub["score"].between(1, 6).all()
        print("Submission file format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    run_demo()
