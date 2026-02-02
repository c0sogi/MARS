import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, alaska_weighted_auc
from library.dataset import AlaskaDataset, get_transforms
from library.model import StegoNet
from library.train import fit
from library.inference import generate_submission


def main():
    print("=== Steganography Detection Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and debugging
    Config.debug = True
    Config.epochs = 1
    Config.debug_train_size = 32  # Small subset for quick training
    Config.debug_val_size = 16  # Small subset for quick validation
    Config.train_batch_size = 4
    Config.val_batch_size = 4

    # Disable downloading pretrained weights for the demo to save time/bandwidth
    # The model will be initialized with random weights, which is fine for pipeline verification
    Config.pretrained = False

    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.seed)

    print(f"Working Directory: {Config.working_dir}")
    print(f"Device: {Config.device}")

    # -------------------------------------------------------------------------
    # 2. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # A. Train Dataset (Paired: Cover + Stego)
    train_ds = AlaskaDataset("train", transform=get_transforms("train"))
    print(
        f"   Train Dataset Size: {len(train_ds)} (Expected: {Config.debug_train_size})"
    )
    assert len(train_ds) == Config.debug_train_size, "Train dataset size mismatch."

    # Fetch one sample
    # Train returns: (images, labels) where images is (2, C, H, W) and labels is (2,)
    images, labels = train_ds[0]
    print(f"   Train Sample Shape: Images={images.shape}, Labels={labels.shape}")

    assert images.shape == (
        2,
        3,
        512,
        512,
    ), "Train images shape incorrect. Expected (2, 3, 512, 512)."
    assert labels.shape == (2,), "Train labels shape incorrect."
    assert (
        labels[0] == 0.0 and labels[1] == 1.0
    ), "Train labels must be [0, 1] (Cover, Stego)."

    # B. Validation Dataset (Single: Image + Label)
    val_ds = AlaskaDataset("val", transform=get_transforms("val"))
    img, label = val_ds[0]
    print(f"   Val Sample Shape: Image={img.shape}, Label={label}")
    assert img.shape == (3, 512, 512), "Val image shape incorrect."
    assert isinstance(label.item(), float), "Val label should be a float."

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = StegoNet(backbone_name=Config.backbone_name, pretrained=False)
    model.to(Config.device)
    model.eval()

    # Create dummy input batch (Batch Size = 2)
    dummy_input = torch.randn(2, 3, 512, 512).to(Config.device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Model Output Shape: {output.shape}")
    # Output should be (Batch_Size, 1) -> Logits
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    # -------------------------------------------------------------------------
    # 4. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Weighted AUC Metric...")

    # Case 1: Perfect Prediction
    y_true = np.array([0, 0, 1, 1])
    # Logits: negative for 0, positive for 1
    y_logits_perfect = np.array([-5.0, -3.0, 3.0, 5.0])
    auc_perfect = alaska_weighted_auc(y_true, y_logits_perfect)
    print(f"   Perfect AUC Score: {auc_perfect}")
    assert np.isclose(auc_perfect, 1.0), "Perfect predictions should yield AUC 1.0"

    # Case 2: Worst Prediction
    y_logits_worst = np.array([5.0, 3.0, -3.0, -5.0])
    auc_worst = alaska_weighted_auc(y_true, y_logits_worst)
    print(f"   Worst AUC Score:   {auc_worst}")
    assert np.isclose(auc_worst, 0.0), "Inverted predictions should yield AUC 0.0"

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Fit)...")

    # Run the fit function. This handles data loading, training, validation,
    # scheduling, and checkpoint saving.
    fit(epochs=Config.epochs, batch_size=Config.train_batch_size, debug=True)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"   Success: Checkpoint found at {best_model_path}")
    else:
        raise FileNotFoundError("Training finished but 'best_model.pth' was not found.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Generate predictions using the model trained in step 5
    generate_submission(checkpoint_path=best_model_path, debug=True)

    # Verify submission file
    if os.path.exists(Config.submission_path):
        df_sub = pd.read_csv(Config.submission_path)
        print(f"   Submission saved to {Config.submission_path}")
        print(f"   Submission Shape: {df_sub.shape}")
        print("   First 3 rows:")
        print(df_sub.head(3))

        # Validation
        assert list(df_sub.columns) == ["Id", "Label"], "Submission columns mismatch."
        assert len(df_sub) > 0, "Submission file is empty."
        # In debug mode, inference runs on a subset of the test set
        expected_len = min(5000, Config.debug_val_size)
        assert (
            len(df_sub) == expected_len
        ), f"Expected {expected_len} predictions in debug mode."
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
