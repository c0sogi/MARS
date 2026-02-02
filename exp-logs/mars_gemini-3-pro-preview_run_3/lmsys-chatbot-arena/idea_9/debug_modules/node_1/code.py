import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import get_logger
from library.data import get_dataloaders
from library.model import SiameseDebertaModel
from library.engine import train_one_epoch, validate, predict_with_tta


def run_demo():
    print(">>> [1/6] Setting up Demo Configuration...")

    # 1. Configuration Overrides for Speed and Demonstration
    # We modify the Config class attributes directly to affect all modules
    seed_everything(42)

    # Use a lightweight model for the demo to ensure execution finishes in seconds/minutes
    Config.model_name = "microsoft/deberta-v3-xsmall"

    # Enable debug mode to use a tiny subset of data
    Config.debug = True
    Config.debug_subset_size = 20

    # Training hyperparameters for demo
    Config.epochs = 1
    Config.train_batch_size = 2
    Config.valid_batch_size = 4
    Config.gradient_accumulation_steps = 1

    # Paths
    Config.working_dir = "./working/demo_run"
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(
        Config.working_dir, "submission/submission.csv"
    )

    # Create directories
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Initialize Logger
    logger = get_logger("demo", log_file=os.path.join(Config.working_dir, "demo.log"))
    logger.info("Configuration updated for demo run.")
    logger.info(f"Model: {Config.model_name}, Debug Size: {Config.debug_subset_size}")

    # 2. Data Loading
    print(">>> [2/6] Loading Data and Creating DataLoaders...")
    # load_cached_data=False ensures we process the CSVs fresh for this demo run
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=Config.debug, debug_size=Config.debug_subset_size
    )

    # Assertions to verify data loading
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Val loader should not be empty."

    # Inspect a single batch
    batch = next(iter(train_loader))
    expected_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "features",
        "target",
    ]
    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    logger.info(f"Batch inspection passed. Input shape: {batch['input_ids_a'].shape}")

    # 3. Model Initialization
    print(">>> [3/6] Initializing Model...")
    device = Config.device
    model = SiameseDebertaModel()
    model.to(device)

    # Verify model components
    assert hasattr(model, "backbone"), "Model missing backbone."
    assert hasattr(model, "pooler"), "Model missing pooler."
    logger.info("Model initialized successfully.")

    # 4. Training Loop Demonstration
    print(">>> [4/6] Running Training (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_fp16)

    # Run one epoch
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, scaler, epoch=0
    )

    logger.info(f"Training Epoch 0 Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # 5. Validation Demonstration
    print(">>> [5/6] Running Validation...")
    val_ce_loss, val_log_loss = validate(model, val_loader, device)

    logger.info(f"Validation CE Loss: {val_ce_loss:.4f}")
    logger.info(f"Validation Log Loss: {val_log_loss:.4f}")
    assert val_log_loss >= 0, "Log loss must be non-negative."

    # Save the model (required for the inference step in the original pipeline)
    torch.save(model.state_dict(), Config.model_save_path)
    logger.info("Model saved.")

    # 6. Inference and Submission
    print(">>> [6/6] Running Inference (TTA) and Generating Submission...")

    # Load the saved model to verify serialization works
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    # Predict
    predictions = predict_with_tta(model, test_loader, device)

    # Assertions on predictions
    expected_rows = len(test_loader.dataset)
    assert predictions.shape == (
        expected_rows,
        3,
    ), f"Shape mismatch: {predictions.shape} vs {(expected_rows, 3)}"

    # Verify probabilities sum to 1
    row_sums = predictions.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    # Generate Submission File
    # We need the IDs corresponding to the subset used in debug mode.
    # Since get_dataloaders subsets internally, we replicate that slice here.
    test_df_full = pd.read_csv(Config.test_path)
    test_df_subset = test_df_full.iloc[: Config.debug_subset_size]

    submission_df = pd.DataFrame(
        {
            "id": test_df_subset["id"],
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    submission_df.to_csv(Config.submission_path, index=False)

    assert os.path.exists(Config.submission_path), "Submission file not found."
    logger.info(f"Submission saved to {Config.submission_path}")
    print(submission_df.head())

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nERROR: Demo execution failed: {e}")
        raise e
