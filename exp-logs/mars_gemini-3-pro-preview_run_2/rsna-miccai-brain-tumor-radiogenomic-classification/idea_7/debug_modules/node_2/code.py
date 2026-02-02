import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.utils import seed_everything, load_metadata, calculate_roc_auc
from library.dataset import BraTSDataset
from library.model import ModalityGatedEfficientNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting BraTS21 MGMT Prediction Demo ===")

    # 1. Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up environment...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # 2. Verify Metadata Loading
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Metadata Loading...")
    df_train = load_metadata("train")
    print(f"    Training samples loaded: {len(df_train)}")
    assert len(df_train) > 0, "Training metadata should not be empty."

    df_test = load_metadata("test")
    print(f"    Test samples loaded: {len(df_test)}")
    assert len(df_test) > 0, "Test metadata should not be empty."

    # 3. Verify Dataset Logic
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset Logic...")
    # Initialize dataset (this will also trigger ROI cache loading/generation)
    train_dataset = BraTSDataset(split="train", load_cached_data=True)

    # Fetch one sample
    sample_idx = 0
    image, label = train_dataset[sample_idx]

    print(f"    Sample Image Shape: {image.shape}")
    print(f"    Sample Label: {label}")

    # Assertions
    # Expected shape: (12 channels, 256 height, 256 width)
    assert image.shape == (12, 256, 256), f"Expected (12, 256, 256), got {image.shape}"
    assert isinstance(image, torch.Tensor), "Image should be a torch Tensor"
    assert label.shape == (1,), f"Expected label shape (1,), got {label.shape}"

    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    model = ModalityGatedEfficientNet(
        num_classes=1, pretrained=False
    )  # Pretrained=False for speed in demo
    model.to(device)
    model.eval()

    # Create a dummy batch of size 2
    dummy_input = torch.stack([image, image]).to(device)  # Shape: (2, 12, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # 5. Run Training Demo
    # -------------------------------------------------------------------------
    print("\n[5] Running Trainer (1 Epoch)...")
    # Initialize Trainer
    # We use a slightly higher learning rate and minimal epochs for demonstration
    trainer = Trainer(learning_rate=1e-3, device=str(device))

    # Run fit
    # Using batch_size=16 and num_workers=4 to utilize the 12 vCPUs and A100
    trainer.fit(
        epochs=1, batch_size=16, num_workers=4, patience=1, load_cached_data=True
    )

    # Check if best model was saved
    assert os.path.exists(
        trainer.best_model_path
    ), "Best model checkpoint was not saved."
    print("    Training completed and model saved.")

    # 6. Run Inference Demo
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference (TTA)...")
    trainer.predict_with_tta(batch_size=16, num_workers=4, load_cached_data=True)

    # 7. Verify Submission
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Submission File...")
    submission_path = "./submission/submission.csv"

    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"    Submission file found with {len(sub_df)} rows.")
        print("    First 3 rows:")
        print(sub_df.head(3))

        # Validation
        assert "BraTS21ID" in sub_df.columns, "Submission missing BraTS21ID column"
        assert "MGMT_value" in sub_df.columns, "Submission missing MGMT_value column"
        assert len(sub_df) == len(
            df_test
        ), f"Submission row count mismatch. Expected {len(df_test)}, got {len(sub_df)}"
        assert (
            sub_df["MGMT_value"].min() >= 0.0 and sub_df["MGMT_value"].max() <= 1.0
        ), "Probabilities out of bounds"
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
