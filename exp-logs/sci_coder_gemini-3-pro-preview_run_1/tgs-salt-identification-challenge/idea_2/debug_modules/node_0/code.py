import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import provided library modules
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model_arch import ResNet34UNetPlusPlus
from library.losses import BCEDiceLoss
from library.trainer import ModelTrainer
from library.inference import predict_and_submit


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(42)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Define working directories for demo
    work_dir = "./working"
    ckpt_dir = os.path.join(work_dir, "demo_checkpoints")
    sub_dir = os.path.join(work_dir, "demo_submission")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Data Pipeline ---")
    # Use debug=True to load a small subset of data for speed
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0, debug=True)

    # Fetch a single batch
    images, masks = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Mask Shape: {masks.shape}")

    # Assertions
    # Shape: (Batch, Channels=2, Height=128, Width=128)
    # Channels=2 because we stack grayscale image + depth channel
    assert images.shape == (
        4,
        2,
        128,
        128,
    ), f"Expected image shape (4, 2, 128, 128), got {images.shape}"
    assert masks.shape == (
        4,
        1,
        128,
        128,
    ), f"Expected mask shape (4, 1, 128, 128), got {masks.shape}"

    # Value range checks
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images should be normalized to [0, 1]"
    assert set(np.unique(masks.numpy()).tolist()).issubset(
        {0, 1}
    ), "Masks should be binary (0 or 1)"
    print("Data pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Check
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")
    model = ResNet34UNetPlusPlus(in_channels=2, n_classes=1).to(device)

    # Create dummy input
    dummy_input = torch.randn(2, 2, 128, 128).to(device)

    # Forward pass
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")

    # Assertions
    assert dummy_output.shape == (
        2,
        1,
        128,
        128,
    ), f"Expected output shape (2, 1, 128, 128), got {dummy_output.shape}"
    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Test
    # -------------------------------------------------------------------------
    print("\n--- Verifying Loss Function ---")
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # Create dummy targets (binary)
    dummy_target = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)

    # Calculate loss
    loss = criterion(dummy_output, dummy_target)

    print(f"Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n--- Executing Training Loop (1 Epoch) ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=1,  # Run only 1 epoch for demonstration
        patience=1,
        checkpoint_dir=ckpt_dir,
        checkpoint_name="demo_best_model.pth",
    )

    # Run training
    best_loss = trainer.run()

    # Verify checkpoint creation
    ckpt_path = os.path.join(ckpt_dir, "demo_best_model.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint file not found at {ckpt_path}"
    print(f"Training loop finished. Best loss: {best_loss:.4f}")
    print("Checkpoint verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n--- Executing Inference on Test Set ---")

    # Load the best model state
    model.load_state_dict(torch.load(ckpt_path))

    # Generate submission
    # Note: predict_and_submit loads the full test set internally
    predict_and_submit(
        model=model, device=device, output_dir=sub_dir, output_name="submission.csv"
    )

    # Verify submission file
    sub_path = os.path.join(sub_dir, "submission.csv")
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    # Verify submission content format
    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "rle_mask" in df_sub.columns, "Submission missing 'rle_mask' column"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference and submission verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
