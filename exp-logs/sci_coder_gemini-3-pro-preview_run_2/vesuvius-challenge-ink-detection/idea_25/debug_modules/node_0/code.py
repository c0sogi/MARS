import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.data import InkDataset
from library.model import UnifiedSegFormer
from library.engine import Trainer
from library.utils import dice_loss, fbeta_score
from library.inference import run_inference


def run_demo():
    print("--- Starting Vesuvius Ink Detection Demo ---")

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to create a "mini" run.
    print("Configuring demo parameters...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.WORKING_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Ensure clean working state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Demonstration (Train/Val)
    print("\n--- Step 1: Data Loading Verification ---")
    # We limit the dataset size to 4 samples to ensure fast execution.
    # This triggers the loading and caching of volume slabs.
    train_dataset = InkDataset(mode="train", limit_size=4, load_cached_data=True)
    val_dataset = InkDataset(mode="validation", limit_size=4, load_cached_data=True)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Verify item structure
    img, mask = train_dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Assertions
    # Image should be (3, 512, 512) based on Config.IN_CHANNELS=3 and TILE_SIZE=512
    assert img.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Expected image shape (3, {Config.TILE_SIZE}, {Config.TILE_SIZE}), got {img.shape}"
    # Mask should be (1, 512, 512)
    assert mask.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Expected mask shape (1, {Config.TILE_SIZE}, {Config.TILE_SIZE}), got {mask.shape}"
    # Check normalization (approximate)
    assert img.min() >= 0.0 and img.max() <= 1.0, "Image data not normalized to [0, 1]"

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Instantiation and Forward Pass
    print("\n--- Step 2: Model Verification ---")
    model = UnifiedSegFormer()
    model.to(device)

    # Create a dummy batch from the loaded data
    dummy_imgs = img.unsqueeze(0).to(device)  # (1, 3, 512, 512)
    dummy_masks = mask.unsqueeze(0).to(device)  # (1, 1, 512, 512)

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(dummy_imgs)

    print(f"Output Logits Shape: {outputs.shape}")
    assert outputs.shape == (
        1,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Model output shape mismatch."

    # 4. Loss and Metric Calculation
    print("\n--- Step 3: Metric Verification ---")
    loss = dice_loss(outputs, dummy_masks)
    score = fbeta_score(outputs, dummy_masks, beta=0.5)

    print(f"Calculated Dice Loss: {loss.item():.4f}")
    print(f"Calculated F0.5 Score: {score:.4f}")

    assert isinstance(loss.item(), float), "Loss should be a scalar float."
    assert 0 <= score <= 1, "F0.5 score should be between 0 and 1."

    # 5. Training Loop
    print("\n--- Step 4: Training Loop Execution ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
    )

    # Run for 1 epoch
    trainer.fit(epochs=Config.EPOCHS)

    # Verify checkpoint creation
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint was not saved."
    print(f"Model successfully saved to {Config.BEST_MODEL_PATH}")

    # 6. Inference Pipeline
    print("\n--- Step 5: Inference Pipeline Execution ---")
    # We use the run_inference helper which handles loading, prediction, and submission generation.
    # We limit size to ensure it runs quickly on the test set.

    try:
        run_inference(
            checkpoint_path=Config.BEST_MODEL_PATH,
            batch_size=Config.BATCH_SIZE,
            limit_size=2,  # Process only 2 test tiles
        )
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file generated with {len(df_sub)} rows.")
        print(df_sub.head())

        # Basic format check
        assert (
            "Id" in df_sub.columns and "Predicted" in df_sub.columns
        ), "Submission file missing required columns."
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
