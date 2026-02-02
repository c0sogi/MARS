import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging

# Import from the provided library
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.data import get_loaders, get_test_loader
from library.modeling import get_model
from library.training import train_one_epoch, valid_one_epoch, inference_fn
from library.soup_utils import average_weights


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print(">>> Setting up configuration and seed...")
    seed_everything(CFG.seed)

    # Override CFG for a quick demo run
    CFG.debug = True  # Use small subset of data
    CFG.epochs = 2  # Run only 2 epochs
    CFG.num_folds = 2  # Reduce folds (we will only run fold 0)
    CFG.batch_size = 8  # Small batch size
    CFG.print_freq = 10
    CFG.output_dir = "./working"  # Use working dir for checkpoints

    # Ensure working directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Logger
    logger = get_logger(os.path.join(CFG.output_dir, "demo.log"))
    logger.info("Configuration configured for demo execution.")

    # 2. Data Loading Verification
    print("\n>>> Verifying Data Loaders...")
    # Get loaders for Fold 0
    train_loader, val_loader = get_loaders(fold=0, cfg=CFG)
    test_loader = get_test_loader(CFG)

    # Check Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")
    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.image_size,
        CFG.image_size,
    ), "Incorrect train image shape"
    assert labels.shape == (CFG.batch_size,), "Incorrect train label shape"

    # Check Test Loader
    test_images, test_ids = next(iter(test_loader))
    print(f"Test Batch Shape: Images {test_images.shape}, IDs {test_ids.shape}")
    assert test_images.shape == (
        CFG.batch_size,
        3,
        CFG.image_size,
        CFG.image_size,
    ), "Incorrect test image shape"

    # 3. Model Instantiation
    print("\n>>> Instantiating Model...")
    # Use the first model in the list, pretrained=False for speed/offline safety in demo
    model_name = CFG.model_names[0]
    model = get_model(model_name, pretrained=False, num_classes=1)
    model.to(CFG.device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, CFG.image_size, CFG.image_size).to(CFG.device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\n>>> Running Training Loop (2 Epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Checkpoint paths
    ckpt_path_1 = os.path.join(CFG.output_dir, "model_epoch_1.pth")
    ckpt_path_2 = os.path.join(CFG.output_dir, "model_epoch_2.pth")

    # Epoch 1
    train_loss = train_one_epoch(
        1, model, train_loader, optimizer, criterion, CFG.device
    )
    val_loss, val_preds = valid_one_epoch(1, model, val_loader, criterion, CFG.device)
    print(f"Epoch 1: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save Checkpoint 1
    torch.save({"state_dict": model.state_dict()}, ckpt_path_1)

    # Epoch 2
    train_loss = train_one_epoch(
        2, model, train_loader, optimizer, criterion, CFG.device
    )
    val_loss, val_preds = valid_one_epoch(2, model, val_loader, criterion, CFG.device)
    print(f"Epoch 2: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save Checkpoint 2
    torch.save({"state_dict": model.state_dict()}, ckpt_path_2)

    assert os.path.exists(ckpt_path_1) and os.path.exists(
        ckpt_path_2
    ), "Checkpoints not saved"

    # 5. Model Soup Verification
    print("\n>>> Verifying Model Soup (Weight Averaging)...")
    # Load individual weights to verify calculation later
    state_1 = torch.load(ckpt_path_1, map_location="cpu")["state_dict"]
    state_2 = torch.load(ckpt_path_2, map_location="cpu")["state_dict"]

    # Perform Soup
    soup_state_dict = average_weights([ckpt_path_1, ckpt_path_2])

    # Verify a random parameter (e.g., the first weight tensor found)
    param_key = list(soup_state_dict.keys())[0]
    val_1 = state_1[param_key].float()
    val_2 = state_2[param_key].float()
    soup_val = soup_state_dict[param_key].float()

    expected_val = (val_1 + val_2) / 2.0

    # Check if values are close (handling potential float precision issues)
    diff = torch.abs(soup_val - expected_val).max().item()
    print(f"Max difference between calculated soup and expected average: {diff:.8f}")
    assert torch.allclose(
        soup_val, expected_val, atol=1e-6
    ), "Soup averaging failed logic check"

    # Load soup weights into model
    model.load_state_dict(soup_state_dict)
    print("Soup weights loaded successfully.")

    # 6. Inference and Submission
    print("\n>>> Running Inference on Test Set...")
    # Run inference
    preds = inference_fn(model, test_loader, CFG.device)

    print(f"Predictions Shape: {preds.shape}")
    assert preds.shape[0] == len(
        test_loader.dataset
    ), "Number of predictions does not match test set size"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions are not valid probabilities"

    # Create Submission DataFrame
    test_df = pd.read_csv(CFG.test_csv)
    if CFG.debug:
        # In debug mode, data.py samples the test set, so we must align the dataframe
        test_df = test_df.sample(n=100, random_state=CFG.seed).reset_index(drop=True)

    submission = pd.DataFrame({"id": test_df["id"], "label": preds})

    submission_path = os.path.join(CFG.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("First 5 rows:")
    print(submission.head())

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
