import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MultiTaskEfficientNet, probabilistic_f1
from library.train import train_model
from library.inference import run_inference


def main():
    print("Starting demonstration script...")

    # 1. Configuration Override for Speed and Debugging
    # We modify the Config class attributes directly to affect the library modules
    print("Configuring for fast execution...")
    Config.DEBUG = True  # Use a small subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 2. Metric Verification
    print("\n--- Verifying Metric (pF1) ---")
    # Case 1: Perfect prediction
    probs_perfect = np.array([0.1, 0.9, 0.1, 0.9])
    targets_perfect = np.array([0, 1, 0, 1])
    # pTP = 1.8, Sum(p)=2.0, Sum(y)=2.0 -> pPrec=0.9, pRec=0.9 -> F1 ~ 0.9
    pf1_perfect = probabilistic_f1(probs_perfect, targets_perfect)
    print(f"pF1 (Good Case): {pf1_perfect:.4f}")
    assert pf1_perfect > 0.8, "pF1 calculation seems incorrect for good predictions."

    # Case 2: Bad prediction
    probs_bad = np.array([0.9, 0.1, 0.9, 0.1])
    targets_bad = np.array([0, 1, 0, 1])
    pf1_bad = probabilistic_f1(probs_bad, targets_bad)
    print(f"pF1 (Bad Case): {pf1_bad:.4f}")
    assert pf1_bad < 0.2, "pF1 calculation seems incorrect for bad predictions."
    print("Metric verification passed.")

    # 3. Data Pipeline Verification
    print("\n--- Verifying Data Pipeline ---")
    # This will load metadata, create datasets, and return loaders
    # Since DEBUG=True, it uses a small sample.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    inputs, cancer_targets, density_targets = next(iter(train_loader))

    print(f"Input Batch Shape: {inputs.shape}")
    print(f"Cancer Targets Shape: {cancer_targets.shape}")
    print(f"Density Targets Shape: {density_targets.shape}")

    # Assertions
    # Inputs: [Batch, Channels, Height, Width]
    # Channels should be 3 (Image + Age + Implant)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Expected input shape {(Config.BATCH_SIZE, Config.IN_CHANNELS, Config.IMG_SIZE[0], Config.IMG_SIZE[1])}, got {inputs.shape}"

    # Targets
    assert (
        cancer_targets.shape[0] == Config.BATCH_SIZE
    ), "Cancer target batch size mismatch"
    assert (
        density_targets.shape[0] == Config.BATCH_SIZE
    ), "Density target batch size mismatch"

    # Check value ranges
    # Image is [0,1], Age is standardized (can be negative), Implant is 0 or 1.
    assert (
        inputs.max() <= 10.0 and inputs.min() >= -10.0
    ), "Input normalization check failed."

    print("Data pipeline verification passed.")

    # 4. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    device = torch.device(Config.DEVICE)
    # Using pretrained=False for speed in this demo, though training usually uses True
    model = MultiTaskEfficientNet(backbone_name=Config.BACKBONE, pretrained=False)
    model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        inputs = inputs.to(device)
        cancer_logits, density_logits = model(inputs)

    print(f"Cancer Logits Shape: {cancer_logits.shape}")
    print(f"Density Logits Shape: {density_logits.shape}")

    assert cancer_logits.shape == (Config.BATCH_SIZE, 1), "Cancer logits shape mismatch"
    assert density_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_AUX_CLASSES,
    ), "Density logits shape mismatch"
    print("Model architecture verification passed.")

    # 5. Training Loop Execution
    print("\n--- Executing Training Loop (1 Epoch, Debug) ---")
    # We use the library's train_model function which handles the loop, validation, and saving
    train_model(
        epochs=Config.EPOCHS,
        debug=Config.DEBUG,
        load_cached_data=True,  # Use the cache generated in step 3
        save_path=Config.MODEL_SAVE_PATH,
        submission_path=Config.SUBMISSION_PATH,
    )

    # Verify model file was created
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training execution passed.")

    # 6. Inference Execution
    print("\n--- Executing Inference ---")
    # Run inference using the saved model
    run_inference(
        checkpoint_path=Config.MODEL_SAVE_PATH,
        submission_path=Config.SUBMISSION_PATH,
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    expected_cols = ["prediction_id", "cancer"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
    assert len(df_sub) > 0, "Submission file is empty."

    print("Inference execution passed.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
