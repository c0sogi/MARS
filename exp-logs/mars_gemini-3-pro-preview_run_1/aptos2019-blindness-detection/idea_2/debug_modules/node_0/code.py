import os
import sys
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa, load_checkpoint
from library.dataset import create_dataloaders
from library.model import OrdinalEfficientNet
from library.engine import train_model, generate_submission

if __name__ == "__main__":
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # -------------------------------------------------------------------------
    print("[1/7] Configuring environment for speed...")
    # Enable debug mode to use a small subset of data (defined in Config.debug_sample_size)
    Config.debug = True
    Config.debug_sample_size = 32  # Use only 32 images

    # Reduce image size for faster processing
    Config.image_size = 256

    # Adjust batch size and workers
    Config.batch_size = 8
    Config.num_workers = 2

    # Use a lightweight backbone without pre-trained weights for speed/offline capability
    Config.backbone = "resnet18"
    Config.pretrained = False

    # Run for only 1 epoch
    Config.epochs = 1

    # Ensure working directory exists (Config.setup() runs on import, but good to double check)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set random seed
    seed_everything(Config.seed)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2/7] Verifying Data Pipeline...")
    train_loader, val_loader, test_loader = create_dataloaders()

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Verify Image Tensor
    # Shape: (Batch, Channels, Height, Width)
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), f"Expected image shape {(Config.batch_size, 3, Config.image_size, Config.image_size)}, got {images.shape}"

    # Verify Target Tensor
    # Shape: (Batch, Num_Ordinal_Units) -> (8, 4) for 5 classes
    assert targets.shape == (
        Config.batch_size,
        Config.num_ordinal_units,
    ), f"Expected target shape {(Config.batch_size, Config.num_ordinal_units)}, got {targets.shape}"

    print("Data pipeline verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3/7] Verifying Model Architecture...")
    device = Config.device

    # Initialize model
    model = OrdinalEfficientNet(
        backbone_name=Config.backbone, pretrained=Config.pretrained
    )
    model.to(device)

    # Run forward pass
    with torch.no_grad():
        # Move inputs to device
        dummy_input = images.to(device)
        outputs = model(dummy_input)

    print(f"Output shape: {outputs.shape}")

    # Verify Output Shape
    assert outputs.shape == (
        Config.batch_size,
        Config.num_ordinal_units,
    ), "Model output shape mismatch."

    # Verify Output Range (Sigmoid should be [0, 1])
    assert (outputs >= 0).all() and (
        outputs <= 1
    ).all(), "Model outputs are not valid probabilities (must be between 0 and 1)."

    print("Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4/7] Executing Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs
    )

    # Train
    best_qwk = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        patience=1,
    )

    print(f"Training finished. Best Validation QWK: {best_qwk}")

    # -------------------------------------------------------------------------
    # 5. Checkpoint Verification
    # -------------------------------------------------------------------------
    print("\n[5/7] Verifying Checkpoint Saving...")
    checkpoint_path = os.path.join(Config.working_dir, "best_model.pth")

    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file was not found at {checkpoint_path}"

    print("Checkpoint found.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6/7] Verifying Inference and Submission...")

    # Re-initialize model to ensure we are loading weights correctly
    inference_model = OrdinalEfficientNet(
        backbone_name=Config.backbone, pretrained=Config.pretrained
    )
    inference_model.to(device)

    # Load weights
    start_epoch, loaded_score = load_checkpoint(
        inference_model, filename="best_model.pth"
    )
    print(f"Loaded model from epoch {start_epoch} with score {loaded_score}")

    # Generate submission
    generate_submission(inference_model, test_loader, device)

    # Verify CSV
    submission_path = Config.submission_path
    assert os.path.exists(submission_path), "Submission CSV not found."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    assert (
        "id_code" in df_sub.columns and "diagnosis" in df_sub.columns
    ), "Submission CSV missing required columns."

    # In debug mode, test_loader only has `debug_sample_size` items
    assert (
        len(df_sub) == Config.debug_sample_size
    ), f"Expected {Config.debug_sample_size} predictions, found {len(df_sub)}"

    print("Inference verified successfully.")

    # -------------------------------------------------------------------------
    # 7. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n[7/7] Verifying Metric Calculation...")

    # Perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred_perfect = np.array([0, 1, 2, 3, 4])
    score_perfect = quadratic_weighted_kappa(y_true, y_pred_perfect)

    # Complete disagreement (inverse)
    y_pred_bad = np.array([4, 3, 2, 1, 0])
    score_bad = quadratic_weighted_kappa(y_true, y_pred_bad)

    print(f"Perfect Score: {score_perfect}")
    print(f"Bad Score: {score_bad}")

    assert np.isclose(score_perfect, 1.0), "Metric failed on perfect agreement."
    assert score_bad < 1.0, "Metric failed on disagreement."

    print("Metric calculation verified.")

    print("\n=== Demonstration Complete: All checks passed. ===")
