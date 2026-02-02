import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission, calculate_roc_auc
from library.dataset import load_data, BirdDataset, MixupCollate
from library.models import BirdClassifier
from library.sam import SAM
from library.engine import train_fn, eval_fn, inference_fn


def main():
    print("Starting Library Usage Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config global state to run a fast debug pass
    set_seed(42)

    print("Configuring for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.TOTAL_STEPS = 2  # Minimal steps
    Config.TTA_SHIFTS = [0.0, 0.5]  # Reduced TTA shifts for demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading Verification
    print("\n--- Testing Data Loading ---")
    # Load training data (debug subset)
    images, labels, rec_ids = load_data(Config.TRAIN_CSV, split="train")

    # Assertions to verify data loading
    assert isinstance(images, np.ndarray), "Images should be a numpy array"
    assert isinstance(labels, np.ndarray), "Labels should be a numpy array"
    assert (
        len(images) == len(labels) == len(rec_ids)
    ), "Data arrays must have same length"
    assert len(images) <= Config.DEBUG_SUBSET_SIZE, "Should respect debug subset size"
    assert images.shape[1:] == (
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
        3,
    ), f"Unexpected image shape: {images.shape}"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Unexpected label shape: {labels.shape}"

    print(f"Successfully loaded {len(images)} samples.")
    print(f"Image shape: {images.shape}, Label shape: {labels.shape}")

    # 3. Dataset and DataLoader Verification
    print("\n--- Testing Dataset and MixupCollate ---")
    dataset = BirdDataset(images, labels, rec_ids, split="train")

    # Test __getitem__
    img_tensor, label_tensor, rec_id = dataset[0]
    assert isinstance(img_tensor, torch.Tensor), "Dataset should return tensors"
    assert img_tensor.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Incorrect tensor channel/dim order"

    # Test DataLoader with Mixup
    collate_fn = MixupCollate(alpha=0.4)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    # Fetch one batch
    batch_images, batch_labels, batch_ids = next(iter(loader))

    assert batch_images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    )
    assert batch_labels.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    print("DataLoader and MixupCollate functioning correctly.")

    # 4. Model Initialization Verification
    print("\n--- Testing Model Initialization ---")
    device = Config.DEVICE
    # Use resnet18 for speed
    model = BirdClassifier(backbone_name="resnet18")
    model.to(device)

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert outputs.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {outputs.shape}"
    print("Model initialized and forward pass successful.")

    # 5. Optimizer (SAM) Setup
    print("\n--- Testing SAM Optimizer Setup ---")
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, rho=0.05)

    # Verify SAM specific methods exist
    assert hasattr(optimizer, "first_step"), "SAM optimizer missing first_step"
    assert hasattr(optimizer, "second_step"), "SAM optimizer missing second_step"
    print("SAM Optimizer initialized.")

    # 6. Training Loop Verification
    print("\n--- Testing Training Loop (train_fn) ---")
    criterion = nn.BCEWithLogitsLoss()

    # Run training for one 'epoch' (iteration over the small loader)
    avg_loss = train_fn(
        model, loader, optimizer, device, scheduler=None, criterion=criterion
    )

    assert isinstance(avg_loss, float), "train_fn should return a float loss"
    assert avg_loss > 0, "Loss should be positive"
    print(f"Training loop executed. Average Loss: {avg_loss:.4f}")

    # 7. Evaluation Verification
    print("\n--- Testing Evaluation Loop (eval_fn) ---")
    # Use the same loader for validation demo
    val_loss, val_auc, val_preds, val_labels = eval_fn(model, loader, device, criterion)

    assert isinstance(val_loss, float)
    assert isinstance(val_auc, float)
    assert val_preds.shape == (len(images), Config.NUM_CLASSES)
    # AUC might be 0.0 if only one class is present in the small subset, which is valid logic-wise
    print(f"Eval loop executed. Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 8. Inference & TTA Verification
    print("\n--- Testing Inference with TTA (inference_fn) ---")
    # Pass numpy images directly as per function signature
    test_rec_ids, test_probs = inference_fn(model, images, rec_ids, device)

    assert len(test_rec_ids) == len(rec_ids)
    assert test_probs.shape == (len(images), Config.NUM_CLASSES)
    assert np.all(
        (test_probs >= 0) & (test_probs <= 1)
    ), "Probabilities must be between 0 and 1"
    print("Inference with TTA executed successfully.")

    # 9. Submission Verification
    print("\n--- Testing Submission Generation ---")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(test_rec_ids, test_probs, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify content format
    df_sub = pd.read_csv(submission_path)
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"
    assert (
        len(df_sub) == len(images) * Config.NUM_CLASSES
    ), "Incorrect number of rows in submission"

    # Check Id format (rec_id * 100 + species_id)
    # Take the first row
    first_id = df_sub.iloc[0]["Id"]
    first_rec_id = test_rec_ids[0]
    expected_start = first_rec_id * 100
    assert (
        expected_start <= first_id < expected_start + Config.NUM_CLASSES
    ), "ID generation logic seems incorrect"

    print(f"Submission saved and verified at {submission_path}")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
