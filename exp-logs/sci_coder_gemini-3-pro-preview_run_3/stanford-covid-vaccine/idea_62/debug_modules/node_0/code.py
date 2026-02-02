import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import RNAModel
from library.utils import MCRMSELoss, calculate_metric
from library.engine import fit, set_seed


def run_demo():
    print("==== Starting RNA Degradation Prediction Demo ====")

    # 1. Override Configuration for Speed and Demo Isolation
    print("\n[Step 1] Configuring environment...")

    # Use a separate working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Modify Config attributes at runtime
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples

    # Reduce Model capacity for instant initialization and forward pass
    Config.HIDDEN_DIM = 32  # Default is 384
    Config.CONV_FILTERS = 32  # Default is 256

    # Training settings for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Data Loading and Verification
    print("\n[Step 2] Loading Data...")
    # load_cached_data=False forces reprocessing of the subset defined by DEBUG_SUBSET_SIZE
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print("Verifying Data Shapes...")
    sample_batch = next(iter(train_loader))
    inputs = sample_batch["inputs"].to(Config.DEVICE)
    bpp_indices = sample_batch["bpp_indices"].to(Config.DEVICE)
    bpp_masks = sample_batch["bpp_masks"].to(Config.DEVICE)
    targets = sample_batch["targets"].to(Config.DEVICE)
    ids = sample_batch["id"]

    # Assertions for Data Shapes
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {inputs.shape}"
    # BPP Indices: (Batch, Seq_Len=107)
    assert bpp_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"BPP Indices shape mismatch: {bpp_indices.shape}"
    # Targets: (Batch, Seq_Scored=68, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {targets.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization and Forward Pass
    print("\n[Step 3] Initializing Model...")
    model = RNAModel().to(Config.DEVICE)

    print("Running Forward Pass...")
    outputs = model(inputs, bpp_indices, bpp_masks)

    # Assertions for Output Shapes
    # Model outputs predictions for full sequence length (107), even though targets are 68
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Output shape mismatch: {outputs.shape}"

    print(f"Model Output Shape: {outputs.shape} (Verified)")

    # 4. Loss and Metric Verification
    print("\n[Step 4] Verifying Loss and Metric...")
    loss_fn = MCRMSELoss()

    # Calculate Loss
    loss = loss_fn(outputs, targets)
    print(f"Calculated MCRMSE Loss: {loss.item():.4f}")

    # Assert Loss is valid
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Calculate Validation Metric (uses numpy, handles slicing)
    metric_score = calculate_metric(outputs, targets)
    print(f"Calculated Validation Metric: {metric_score:.4f}")
    assert isinstance(metric_score, float), "Metric should return a float"

    # 5. Training Loop
    print("\n[Step 5] Running Training Loop...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Verify Model Saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved!"
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # 6. Inference Demonstration
    print("\n[Step 6] Running Inference on Test Set...")

    # Load best model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    test_batch = next(iter(test_loader))
    t_inputs = test_batch["inputs"].to(Config.DEVICE)
    t_bpp_indices = test_batch["bpp_indices"].to(Config.DEVICE)
    t_bpp_masks = test_batch["bpp_masks"].to(Config.DEVICE)

    with torch.no_grad():
        test_preds = model(t_inputs, t_bpp_indices, t_bpp_masks)

    print(f"Test Predictions Shape: {test_preds.shape}")

    # Assertions for Test Output
    # Should be (Batch, 107, 5)
    assert (
        test_preds.shape[1] == Config.SEQ_LEN
    ), "Test predictions must cover full sequence length"
    assert (
        test_preds.shape[2] == Config.NUM_TARGETS
    ), "Test predictions must have 5 target columns"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
