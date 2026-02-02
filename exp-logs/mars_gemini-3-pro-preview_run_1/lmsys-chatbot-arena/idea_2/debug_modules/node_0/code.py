import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.features import prepare_scalar_features
from library.dataset import build_datasets
from library.model import SiameseDeberta
from library.engine import run_training, run_inference


def main():
    # ------------------------------------------------------------------------
    # 1. Setup and Configuration
    # ------------------------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(42)

    # Initialize logger
    logger = setup_logger("demo_execution")
    logger.info("Starting demonstration script...")

    # Override Config for fast execution (Debug Mode)
    logger.info("Configuring parameters for fast execution...")
    Config.debug = True  # Uses subset: 100 train, 50 val, 50 test
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4

    # Set up a specific working directory for this demo to avoid path conflicts
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths to point to the demo directory
    Config.working_dir = demo_dir
    Config.model_save_path = os.path.join(demo_dir, "siamese_model_demo.pth")
    Config.submission_path = os.path.join(demo_dir, "submission_demo.csv")
    Config.train_features_path = os.path.join(demo_dir, "train_scalar_features.parquet")
    Config.val_features_path = os.path.join(demo_dir, "val_scalar_features.parquet")
    Config.test_features_path = os.path.join(demo_dir, "test_scalar_features.parquet")

    # ------------------------------------------------------------------------
    # 2. Feature Engineering
    # ------------------------------------------------------------------------
    logger.info("Step 1: Feature Engineering")

    # Force re-computation (load_cached_data=False) to demonstrate the logic
    # This will read metadata, compute features, normalize, and save to parquet
    train_feats, val_feats, test_feats = prepare_scalar_features(load_cached_data=False)

    # Validation
    # Debug mode should yield 100 training samples and 50 val/test samples
    assert len(train_feats) == 100, f"Expected 100 train feats, got {len(train_feats)}"
    assert len(val_feats) == 50, f"Expected 50 val feats, got {len(val_feats)}"
    assert len(test_feats) == 50, f"Expected 50 test feats, got {len(test_feats)}"

    # Check feature dimensions
    assert (
        train_feats.shape[1] == Config.num_scalar_features
    ), f"Expected {Config.num_scalar_features} features, got {train_feats.shape[1]}"

    logger.info("Feature engineering verification passed.")

    # ------------------------------------------------------------------------
    # 3. Dataset Creation
    # ------------------------------------------------------------------------
    logger.info("Step 2: Dataset Creation")

    # Build datasets (this will internally load the features we just created)
    train_ds, val_ds, test_ds = build_datasets(load_cached_data=True)

    # Validation
    assert len(train_ds) == 100
    assert len(val_ds) == 50
    assert len(test_ds) == 50

    # Inspect one item from the training dataset
    sample_item = train_ds[0]
    expected_keys = {
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalar_features",
        "label",
    }
    assert expected_keys.issubset(sample_item.keys()), "Missing keys in dataset item"

    # Check tensor shapes
    assert (
        sample_item["input_ids_a"].shape[0] == Config.max_length
    ), "Incorrect sequence length"
    assert sample_item["label"].shape == (
        3,
    ), "Label should be a 3-class probability vector"

    logger.info("Dataset verification passed.")

    # ------------------------------------------------------------------------
    # 4. DataLoader Setup
    # ------------------------------------------------------------------------
    logger.info("Step 3: DataLoader Setup")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.valid_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.valid_batch_size, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------------
    # 5. Model Initialization & Forward Pass Check
    # ------------------------------------------------------------------------
    logger.info("Step 4: Model Initialization")

    device = Config.device
    model = SiameseDeberta()
    model.to(device)

    # Run a dummy forward pass with one batch to verify architecture
    batch = next(iter(train_loader))
    with torch.no_grad():
        logits = model(
            batch["input_ids_a"].to(device),
            batch["attention_mask_a"].to(device),
            batch["input_ids_b"].to(device),
            batch["attention_mask_b"].to(device),
            batch["scalar_features"].to(device),
        )

    # Check output shape: (Batch Size, Num Classes)
    assert logits.shape == (
        Config.train_batch_size,
        3,
    ), f"Logits shape mismatch. Expected ({Config.train_batch_size}, 3), got {logits.shape}"

    logger.info("Model forward pass verification passed.")

    # ------------------------------------------------------------------------
    # 6. Training Loop
    # ------------------------------------------------------------------------
    logger.info("Step 5: Training Loop")

    # Run training (1 epoch as configured)
    run_training(model, train_loader, val_loader, epochs=Config.epochs, device=device)

    # Verify that the model checkpoint was saved
    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError(f"Model file not found at {Config.model_save_path}")

    logger.info("Training verification passed.")

    # ------------------------------------------------------------------------
    # 7. Inference
    # ------------------------------------------------------------------------
    logger.info("Step 6: Inference")

    # Run inference on test set
    run_inference(model, test_loader, device=device)

    # Verify submission file
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    # Check submission content
    sub_df = pd.read_csv(Config.submission_path)

    # In debug mode, we expect 50 predictions
    assert len(sub_df) == 50, f"Expected 50 predictions, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Check values are probabilities (roughly)
    # Note: Softmax ensures they sum to 1, but we just check range here
    preds = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].values
    assert (preds >= 0).all() and (
        preds <= 1.0001
    ).all(), "Predictions out of probability range"

    logger.info("Inference verification passed.")
    logger.info("All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
