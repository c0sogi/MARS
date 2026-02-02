import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from torch.optim import AdamW
from transformers import logging as hf_logging

# Import from the provided library files
from library.config import Config
from library.data import prepare_data, get_dataloaders
from library.model import ToxicityModel
from library.trainer import Trainer
from library.utils import seed_everything


# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
def setup_environment():
    # Suppress warnings and logs for clean output
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


class DemoConfig(Config):
    # Override paths to avoid conflicts
    WORKING_DIR = "./working/demo_submission"
    SUBMISSION_DIR = "./working/demo_submission/output"

    # Optimization for speed
    MODEL_NAME = "roberta-base"  # Use base model for faster loading/inference in demo
    MAX_LEN = 64  # Short sequence length
    BATCH_SIZE = 8  # Small batch size
    EPOCHS = 1  # Single epoch

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def slice_data_dictionary(data, n_train=200, n_val=50, n_test=50):
    """
    Slices the large data arrays to a small subset to ensure the demo runs quickly.
    """
    sliced_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if key.startswith("train_"):
                sliced_data[key] = value[:n_train]
            elif key.startswith("val_"):
                sliced_data[key] = value[:n_val]
            elif key.startswith("test_"):
                sliced_data[key] = value[:n_test]
            else:
                sliced_data[key] = value
        else:
            sliced_data[key] = value

    print(f"Data sliced for demo: Train={n_train}, Val={n_val}, Test={n_test}")
    return sliced_data


# ------------------------------------------------------------------------------
# 2. Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    setup_environment()

    # Set seed for reproducibility
    seed_everything(42)

    print("=== Jigsaw Toxicity Bias Mitigation Demo ===")

    # Initialize Config
    config = DemoConfig()
    print(f"Working Directory: {config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # Data Loading & Preprocessing
    # --------------------------------------------------------------------------
    print("\n[1/5] Preparing Data...")
    # We load data (this might take a few minutes for the full read/tokenize)
    # Note: In a real scenario, we would rely on the cache. Here we force a load
    # but then slice it immediately.
    try:
        raw_data = prepare_data(config, load_cached_data=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Optimization: Slice data to a tiny subset for the demo run
    data = slice_data_dictionary(raw_data, n_train=128, n_val=64, n_test=64)

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(config, data)

    # Assertion: Check loader size
    assert len(train_loader) > 0, "Train loader is empty"
    print("DataLoaders created successfully.")

    # --------------------------------------------------------------------------
    # Model Initialization
    # --------------------------------------------------------------------------
    print("\n[2/5] Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ToxicityModel(config)
    model.to(device)

    # Verification: Dummy Forward Pass
    dummy_ids = torch.randint(0, 1000, (2, config.MAX_LEN)).to(device)
    dummy_mask = torch.ones((2, config.MAX_LEN)).to(device)

    with torch.no_grad():
        tox_logits, ident_logits = model(dummy_ids, dummy_mask)

    # Assert output shapes
    # tox_logits: [batch_size, 1] (before view) or [batch_size]
    # ident_logits: [batch_size, num_identities]
    assert tox_logits.shape == (
        2,
        1,
    ), f"Unexpected toxicity logits shape: {tox_logits.shape}"
    assert ident_logits.shape == (
        2,
        len(config.IDENTITY_COLUMNS),
    ), f"Unexpected identity logits shape: {ident_logits.shape}"
    print("Model architecture verified.")

    # --------------------------------------------------------------------------
    # Trainer Setup
    # --------------------------------------------------------------------------
    print("\n[3/5] Setting up Trainer...")
    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
    )

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    print("\n[4/5] Starting Training (Demo Subset)...")
    # We pass the validation identities array required for metric calculation
    # Note: This must match the sliced validation set size
    val_identities = data["val_identities"]

    trainer.fit(val_identities, patience=1)

    # --------------------------------------------------------------------------
    # Prediction & Submission
    # --------------------------------------------------------------------------
    print("\n[5/5] Generating Predictions...")
    # Get the IDs corresponding to the sliced test set
    test_ids = data["test_df_ids"][:64]  # Must match n_test in slice_data_dictionary

    submission_df = trainer.predict(test_loader, test_ids)

    # --------------------------------------------------------------------------
    # Final Validation
    # --------------------------------------------------------------------------
    print("\n=== Validating Output ===")

    # Check 1: Submission file exists
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    # Check 2: Submission content
    df_check = pd.read_csv(sub_path)
    assert (
        len(df_check) == 64
    ), f"Submission length mismatch. Expected 64, got {len(df_check)}"
    assert (
        "id" in df_check.columns and "prediction" in df_check.columns
    ), "Missing columns in submission."
    assert (
        df_check["prediction"].min() >= 0.0 and df_check["prediction"].max() <= 1.0
    ), "Predictions out of range [0, 1]."

    print("All checks passed. Demo completed successfully.")
