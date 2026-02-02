import os
import sys
import torch
import pandas as pd
import csv
import logging

# ---------------------------------------------------------
# 1. Suppress Progress Bars (Requirement)
# ---------------------------------------------------------
# We must patch tqdm before importing library modules that use it.
import tqdm


def no_op_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = no_op_tqdm

# ---------------------------------------------------------
# 2. Import Config and Override for Speed
# ---------------------------------------------------------
from library.config import Config

# Override Config for a quick demonstration run
print("Overriding Config for rapid execution...")
Config.MAX_TRAIN_SAMPLES = 200  # Tiny subset for demo
Config.VAL_SAMPLES = 50  # Tiny subset for demo
Config.EPOCHS = 1  # Single epoch
Config.TRAIN_BATCH_SIZE = 8
Config.VAL_BATCH_SIZE = 8
Config.TEST_BATCH_SIZE = 8
Config.LOG_INTERVAL = 5  # Frequent logging for short run
Config.WORKING_DIR = "./working/demo_run"
Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "outputs")
Config.LOCATOR_CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "locator_checkpoints")
Config.FILLER_CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "filler_checkpoints")
Config.BEST_LOCATOR_PATH = os.path.join(
    Config.LOCATOR_CHECKPOINT_DIR, "best_locator.pth"
)
Config.BEST_FILLER_PATH = os.path.join(Config.FILLER_CHECKPOINT_DIR, "best_filler.pth")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Ensure new directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.LOCATOR_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.FILLER_CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 3. Import Library Modules
# ---------------------------------------------------------
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.predictor import Predictor
from library.models import PointerLocator, get_filler_model

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("\n=== Step 1: Data Loading ===")
    # Load data using debug=True to force small subsets (1000 train, 100 val/test)
    # This ensures the demo runs very fast regardless of Config.MAX_TRAIN_SAMPLES logic in loader
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty!")

    print("\n=== Step 2: Model Logic Verification ===")
    # Fetch a single batch to verify shapes and logic
    batch = next(iter(train_loader))

    # Verify Locator Inputs
    loc_input_ids = batch["locator_input_ids"]
    loc_labels = batch["locator_labels"]
    print(f"Locator Input Shape: {loc_input_ids.shape}")
    print(f"Locator Label Shape: {loc_labels.shape}")

    # Instantiate models manually for a quick forward pass check
    device = Config.DEVICE
    locator = PointerLocator(model_name=Config.MODEL_NAME).to(device)
    filler = get_filler_model(model_name=Config.MODEL_NAME).to(device)

    # Move batch to device
    loc_input_ids = loc_input_ids.to(device)
    loc_mask = batch["locator_attention_mask"].to(device)

    # Check Locator Forward Pass
    with torch.no_grad():
        loc_logits = locator(loc_input_ids, loc_mask)

    # Assert output shape is (batch_size, seq_len)
    assert (
        loc_logits.shape == loc_input_ids.shape
    ), f"Locator logits shape mismatch. Expected {loc_input_ids.shape}, got {loc_logits.shape}"
    print("Locator forward pass successful.")

    # Check Filler Forward Pass
    fil_input_ids = batch["filler_input_ids"].to(device)
    fil_mask = batch["filler_attention_mask"].to(device)
    fil_labels = batch["filler_labels"].to(device)

    with torch.no_grad():
        fil_outputs = filler(
            input_ids=fil_input_ids, attention_mask=fil_mask, labels=fil_labels
        )

    assert fil_outputs.loss is not None, "Filler model did not return a loss."
    print("Filler forward pass successful.")

    # Clean up manual models to free memory
    del locator, filler, loc_logits, fil_outputs
    torch.cuda.empty_cache()

    print("\n=== Step 3: Training Phase ===")
    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Run Training (This will train both Locator and Filler for 1 epoch on the small subset)
    trainer.train()

    # Verify checkpoints were created
    if not os.path.exists(Config.BEST_LOCATOR_PATH):
        raise FileNotFoundError(
            f"Locator checkpoint not found at {Config.BEST_LOCATOR_PATH}"
        )
    if not os.path.exists(Config.BEST_FILLER_PATH):
        raise FileNotFoundError(
            f"Filler checkpoint not found at {Config.BEST_FILLER_PATH}"
        )

    print("Training complete and checkpoints saved.")

    print("\n=== Step 4: Inference Phase ===")
    # Initialize Predictor (loads best checkpoints automatically)
    predictor = Predictor()

    # Generate Submission
    predictor.generate_submission(test_loader)

    print("\n=== Step 5: Submission Verification ===")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate CSV format
    try:
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission loaded successfully. Shape: {df_sub.shape}")

        required_cols = ["id", "sentence"]
        if not all(col in df_sub.columns for col in required_cols):
            raise ValueError(
                f"Submission missing required columns. Found: {df_sub.columns}"
            )

        # Check first row content
        first_sent = df_sub.iloc[0]["sentence"]
        if not isinstance(first_sent, str) or len(first_sent) == 0:
            raise ValueError("First sentence in submission is invalid.")

        print("Submission format verification passed.")
        print(f"Example prediction: {first_sent}")

    except Exception as e:
        raise ValueError(f"Failed to validate submission file: {e}")

    print("\nAll steps completed successfully.")
