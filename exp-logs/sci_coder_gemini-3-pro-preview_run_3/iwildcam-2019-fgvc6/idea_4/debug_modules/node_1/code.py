import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_score, make_weighted_sampler
from library.dataset import AnimalDataset, get_transforms
from library.model import AnimalModel
from library.train import run_training
from library.predict import run_inference


def main():
    print("=== Starting Demonstration of Animal Species Classification Pipeline ===")

    # 1. Setup Configuration for Demo
    # We override Config attributes to run a fast, small-scale test.
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up demo configuration in {DEMO_DIR}...")

    # Override Config paths to point to our demo subsets (to be created)
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_METADATA_PATH = os.path.join(DEMO_DIR, "train_subset.csv")
    Config.VAL_METADATA_PATH = os.path.join(DEMO_DIR, "val_subset.csv")
    Config.TEST_METADATA_PATH = os.path.join(DEMO_DIR, "test_subset.csv")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.IMG_SIZE = 128  # Reduce size for faster forward/backward pass in demo

    # Set seed
    seed_everything(Config.SEED)

    # 2. Prepare Data Subsets
    print("Creating data subsets for rapid testing...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets (ensure we have enough for a batch)
    # We take the first N to ensure files likely exist, though random sample is also fine
    train_subset = orig_train.head(20).copy()
    val_subset = orig_val.head(10).copy()
    test_subset = orig_test.head(10).copy()

    # Save subsets
    train_subset.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    val_subset.to_csv(Config.VAL_METADATA_PATH, index=False)
    test_subset.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(
        f"Subsets saved: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # 3. Verify Utils
    print("\n--- Verifying Library Utils ---")

    # Test get_score
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 2, 0, 0]  # Last one is wrong
    score = get_score(y_true, y_pred)
    print(f"Calculated F1 Score: {score:.4f}")
    assert 0.0 <= score <= 1.0, "F1 score out of range"

    # Test Weighted Sampler
    sampler = make_weighted_sampler(train_subset, target_col="Category")
    assert isinstance(
        sampler, torch.utils.data.WeightedRandomSampler
    ), "Failed to create WeightedRandomSampler"
    print("WeightedRandomSampler created successfully.")

    # 4. Verify Dataset
    print("\n--- Verifying Dataset Class ---")

    # Initialize Dataset
    ds = AnimalDataset(train_subset, transforms=get_transforms("train"), mode="train")

    # Check length
    assert len(ds) == len(train_subset), "Dataset length mismatch"

    # Check item retrieval
    img, label = ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")

    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"
    assert label.dtype == torch.long, "Label is not long type"

    # 5. Verify Model
    print("\n--- Verifying Model Architecture ---")

    # Instantiate model (pretrained=False for speed/offline safety in demo, though training uses True)
    model = AnimalModel(pretrained=False)
    model.eval()

    # Dummy Input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model forward pass successful.")

    # 6. Run Training Loop
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # run_training uses the Config paths we updated earlier
    best_f1 = run_training(debug=False, epochs=Config.EPOCHS, patience=1)

    print(f"Training finished. Best F1: {best_f1}")
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not created."
    print("Checkpoint verified.")

    # 7. Run Inference
    print("\n--- Executing Inference ---")

    run_inference(
        test_metadata_path=Config.TEST_METADATA_PATH,
        model_checkpoint=Config.MODEL_CHECKPOINT_PATH,
        submission_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # 8. Validate Submission Format
    print("\n--- Validating Submission Format ---")
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission Shape: {submission_df.shape}")
    assert len(submission_df) == len(test_subset), "Submission row count mismatch"
    assert list(submission_df.columns) == [
        "Id",
        "Predicted",
    ], f"Incorrect columns: {submission_df.columns}"
    assert submission_df["Predicted"].dtype in [
        np.int64,
        int,
    ], "Predicted column is not integer"

    print("Submission format valid.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
