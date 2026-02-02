import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_micro_f1
from library.dataset import get_dataloaders
from library.model import ArtworkResNet
from library.engine import run_training, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of Artwork Attribute Labeling Library ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config parameters to run a fast, small-scale experiment
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 2  # Reduce workers for small data
    Config.EARLY_STOPPING_PATIENCE = 1

    # Redirect outputs to working directory to avoid permission errors
    Config.IDEA_DIR = "./working/demo_experiment"
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    Config.MODEL_SAVE_PATH = os.path.join(Config.IDEA_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "demo_submission.csv")

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("    Configuration updated. Random seed set.\n")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("[2] Verifying utility functions...")

    # Test calculate_micro_f1
    # Scenario: 2 samples, 3 classes
    # Target: [1, 0, 1], [0, 1, 0]
    # Preds (Probs): [0.8, 0.2, 0.9], [0.1, 0.95, 0.3] -> Threshold 0.5 -> [1, 0, 1], [0, 1, 0]
    # Perfect match, F1 should be 1.0
    dummy_targets = np.array([[1, 0, 1], [0, 1, 0]])
    dummy_preds = np.array([[0.8, 0.2, 0.9], [0.1, 0.95, 0.3]])

    f1 = calculate_micro_f1(dummy_preds, dummy_targets, threshold=0.5)
    assert np.isclose(f1, 1.0), f"Expected F1=1.0, got {f1}"

    # Scenario: Complete mismatch
    dummy_preds_wrong = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
    f1_wrong = calculate_micro_f1(dummy_preds_wrong, dummy_targets, threshold=0.5)
    assert np.isclose(f1_wrong, 0.0), f"Expected F1=0.0, got {f1_wrong}"

    print("    Utility functions verified successfully.\n")

    # ------------------------------------------------------------------------
    # 3. Verify Data Loading
    # ------------------------------------------------------------------------
    print("[3] Initializing DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")

    # Fetch a single batch to verify shapes
    images, targets = next(iter(train_loader))

    # Verify Image Shape: (Batch_Size, Channels, Height, Width)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Verify Target Shape: (Batch_Size, Num_Classes)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    # Verify Target Values (Multi-hot encoding should be 0.0 or 1.0)
    unique_vals = torch.unique(targets)
    assert torch.all(
        torch.isin(unique_vals, torch.tensor([0.0, 1.0]))
    ), "Targets contain values other than 0 and 1."

    print("    DataLoaders functioning correctly. Batch shapes verified.\n")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("[4] Instantiating Model...")

    model = ArtworkResNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Pass the dummy batch through the model
    with torch.no_grad():
        images = images.to(Config.DEVICE)
        outputs = model(images)

    # Verify Output Shape: (Batch_Size, Num_Classes)
    assert (
        outputs.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {outputs.shape}"

    print("    Model instantiated and forward pass successful.\n")

    # ------------------------------------------------------------------------
    # 5. Execute Training Loop
    # ------------------------------------------------------------------------
    print("[5] Running Training Loop (1 Epoch)...")

    # run_training handles the loop, saving, and logging
    run_training(train_loader, val_loader)

    # Verify that the model file was created
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"    Training complete. Model saved to {Config.MODEL_SAVE_PATH}")
    else:
        # If validation F1 was 0 and didn't improve over -1 (unlikely with initialized weights but possible),
        # the code might not save. However, best_f1 starts at -1.0, so any F1 >= 0 saves.
        # If it fails, we raise an error.
        raise FileNotFoundError("Model file was not saved after training.")
    print("")

    # ------------------------------------------------------------------------
    # 6. Execute Inference
    # ------------------------------------------------------------------------
    print("[6] Running Inference on Test Set...")

    inference(test_loader)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Inference complete. Submission saved to {Config.SUBMISSION_PATH}")

        # Validate submission format
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        required_columns = {"id", "attribute_ids"}
        assert required_columns.issubset(
            sub_df.columns
        ), f"Submission missing columns. Found {sub_df.columns}"

        print(f"    Submission contains {len(sub_df)} rows.")
        print("    Format verification successful.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
