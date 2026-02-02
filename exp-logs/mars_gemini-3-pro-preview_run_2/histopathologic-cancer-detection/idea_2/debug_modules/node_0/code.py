import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import PathologyDataset, get_transforms
from library.model import get_model
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("--- Starting Implementation Demonstration ---")

    # 1. Modify Configuration for Speed and Demonstration
    print("\n[Step 1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples for speed
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, Samples=20, Epochs=1.")

    # 2. Dataset and Transform Verification
    print("\n[Step 2] Verifying Dataset and Transforms...")

    # Initialize Training Dataset
    train_dataset = PathologyDataset(mode="train", transform=get_transforms("train"))

    # Verify dataset length (should match DEBUG_SAMPLES)
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} samples, got {len(train_dataset)}"

    # Verify item shape
    img, label = train_dataset[0]
    expected_shape = (3, Config.INPUT_SIZE[0], Config.INPUT_SIZE[1])

    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert (
        img.shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a Tensor"

    print(f"Dataset verified. Sample shape: {img.shape}, Label: {label}")

    # 3. Model Verification
    print("\n[Step 3] Verifying Model Architecture...")
    device = Config.DEVICE
    model = get_model(
        pretrained=False
    )  # No need to download weights for this logic check
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.INPUT_SIZE[0], Config.INPUT_SIZE[1]).to(
        device
    )

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape (Batch Size, Num Classes)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # 4. Training Loop Demonstration
    print("\n[Step 4] Demonstrating Training Loop...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple script to avoid multiprocessing overhead
    )

    val_dataset = PathologyDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Trainer
    trainer = Trainer(model, optimizer, train_loader, val_loader, device=device)

    # Run Training
    print("Executing Trainer.fit()...")
    trainer.fit()

    # Verify Checkpoint Creation
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created."
    print(f"Training complete. Checkpoint saved at {Config.MODEL_CHECKPOINT}")

    # 5. Inference Demonstration
    print("\n[Step 5] Demonstrating Inference...")

    # Run Inference
    # Note: This uses the test dataset (subsetted by DEBUG_SAMPLES)
    run_inference(
        checkpoint_path=Config.MODEL_CHECKPOINT,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        device=device,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(df_submission)}")

    # Verify Submission Content
    assert list(df_submission.columns) == [
        "id",
        "label",
    ], f"Invalid columns: {df_submission.columns}"
    assert (
        len(df_submission) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} predictions, got {len(df_submission)}"

    # Verify values are probabilities (0-1) - though TTA might output logits if not careful,
    # let's check the library code logic.
    # Library `predict_with_tta` does: `avg_probs += probs` where `probs = torch.sigmoid(logits)`.
    # So output should be in [0, 1].
    preds = df_submission["label"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions are not valid probabilities [0, 1]"

    print("Inference verified successfully.")
    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
