import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Ensure the library modules can be imported
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint
from library.dataset import RSNADataset, get_transforms, cache_image_paths
from library.model import CervicalSpineModel
from library.loss import WeightedMultiLabelLoss
from library.train import train_one_epoch, validate
from library.inference import predict


def main():
    # --- 1. Configuration & Setup ---
    print("\n=== 1. Configuration & Setup ===")

    # Override Config for speed and demonstration purposes
    Config.debug = True
    Config.epochs = 1
    Config.batch_size = 2
    Config.seq_len = 16  # Reduced from 96 for speed
    Config.image_size = (256, 256)  # Reduced from 384x384
    Config.backbone = "tf_efficientnet_b0_ns"  # Smaller backbone
    Config.num_workers = 0  # Avoid multiprocessing overhead in demo
    Config.working_dir = "./working/demo_execution"
    Config.accum_iter = 1  # No gradient accumulation needed for small batch
    Config.print_freq = 1

    os.makedirs(Config.working_dir, exist_ok=True)

    seed_everything(Config.seed)
    logger = get_logger("demo.log")
    logger.info("Configuration updated for demo execution.")
    logger.info(f"Device: {Config.device}")

    # --- 2. Dataset & DataLoader Demonstration ---
    print("\n=== 2. Dataset & DataLoader Demonstration ===")

    # Load metadata
    train_df = pd.read_csv(Config.train_metadata_path)

    # Use a tiny subset (4 samples)
    subset_df = train_df.head(4).copy()
    logger.info(f"Using subset of {len(subset_df)} samples for demonstration.")

    # Cache paths (this will scan directories or load from existing cache)
    # We force re-scan or load to ensure it works
    paths_map = cache_image_paths(subset_df, "train", load_cached_data=False)

    # Initialize Dataset
    dataset = RSNADataset(
        metadata_df=subset_df,
        image_paths_map=paths_map,
        phase="train",
        transform=get_transforms("train"),
    )

    # Verify __getitem__
    sample_seq, sample_label = dataset[0]

    logger.info(f"Sample Sequence Shape: {sample_seq.shape}")
    logger.info(f"Sample Label Shape: {sample_label.shape}")

    # Assertions
    # Shape: (Seq_Len, Channels, H, W)
    expected_seq_shape = (Config.seq_len, 3, Config.image_size[0], Config.image_size[1])
    # Shape: (8,) -> 7 vertebrae + 1 overall
    expected_label_shape = (8,)

    if sample_seq.shape != expected_seq_shape:
        raise AssertionError(
            f"Expected sequence shape {expected_seq_shape}, got {sample_seq.shape}"
        )

    if sample_label.shape != expected_label_shape:
        raise AssertionError(
            f"Expected label shape {expected_label_shape}, got {sample_label.shape}"
        )

    logger.info("Dataset shapes verified successfully.")

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        drop_last=False,
    )

    # --- 3. Model & Loss Demonstration ---
    print("\n=== 3. Model & Loss Demonstration ===")

    # Initialize Model
    model = CervicalSpineModel()
    model.to(Config.device)
    logger.info(f"Model {Config.backbone} initialized.")

    # Create dummy batch
    dummy_input = sample_seq.unsqueeze(0).to(Config.device)  # (1, Seq, 3, H, W)
    dummy_target = sample_label.unsqueeze(0).to(Config.device)  # (1, 8)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    logger.info(f"Logits Shape: {logits.shape}")

    if logits.shape != (1, 8):
        raise AssertionError(f"Expected logits shape (1, 8), got {logits.shape}")

    # Loss Calculation
    criterion = WeightedMultiLabelLoss()
    loss = criterion(logits, dummy_target)

    logger.info(f"Calculated Loss: {loss.item()}")

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss is NaN or negative.")

    logger.info("Model forward pass and loss calculation verified.")

    # --- 4. Training Loop Simulation ---
    print("\n=== 4. Training Loop Simulation ===")

    # Setup for training
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scaler = GradScaler()

    # Run one epoch of training
    logger.info("Running train_one_epoch...")
    train_loss = train_one_epoch(
        loader=loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        logger=logger,
        accum_iter=Config.accum_iter,
    )
    logger.info(f"Train Loss: {train_loss:.4f}")

    # Run validation
    logger.info("Running validate...")
    val_loss = validate(loader, model, criterion, logger)
    logger.info(f"Validation Loss: {val_loss:.4f}")

    # Save a dummy checkpoint to test saving logic
    save_checkpoint(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "best_loss": val_loss,
        },
        is_best=True,
        filename="last_checkpoint.pth",
    )

    if not os.path.exists(os.path.join(Config.working_dir, "best_model.pth")):
        raise AssertionError("Checkpoint file was not saved.")

    logger.info("Training loop and checkpointing verified.")

    # --- 5. Inference Demonstration ---
    print("\n=== 5. Inference Demonstration ===")

    # The predict function handles dataset loading, model loading (from best_model.pth),
    # and submission file generation.
    # We use debug=True to run on a small subset of the test data.

    logger.info("Running inference pipeline...")
    submission_df = predict(debug=True, load_cached_data=False)

    # Verify Submission
    expected_cols = ["row_id", "fractured"]
    if list(submission_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"
        )

    if submission_df.empty:
        raise AssertionError("Submission DataFrame is empty.")

    # Check if row_id format is correct (UID_Target)
    sample_row_id = submission_df.iloc[0]["row_id"]
    if "_" not in sample_row_id:
        raise AssertionError(f"row_id format seems incorrect: {sample_row_id}")

    logger.info("Inference completed and submission verified.")
    logger.info("All demonstrations passed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
