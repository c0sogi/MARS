import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    seed_everything,
    DEVICE,
    CACHE_DIR,
)
from library.dataset import ContrailDataset, get_transforms
from library.model import IsotropicConvNeXtUNet
from library.loss import HybridLoss
from library.utils import rle_encode, dice_coef
from library.engine import train_one_epoch, validate, inference


def main():
    # 1. Setup and Reproducibility
    seed_everything(42)
    print("Configuration and seeding complete.")

    # 2. Dataset and DataLoader Demonstration
    print("\n--- Testing Dataset and DataLoader ---")
    # Initialize dataset with a small sample size for speed
    train_dataset = ContrailDataset(
        split="train",
        transform=get_transforms("train"),
        max_samples=4,  # Limit to 4 samples for demonstration
        load_cached_data=True,
    )

    # Verify dataset item structure
    img, mask = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")

    # Assertions for shape correctness
    # Image: (8 channels, 256 height, 256 width)
    # Mask: (1 channel, 256 height, 256 width)
    assert img.shape == (
        8,
        256,
        256,
    ), f"Expected image shape (8, 256, 256), got {img.shape}"
    assert mask.shape == (
        1,
        256,
        256,
    ), f"Expected mask shape (1, 256, 256), got {mask.shape}"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Use 0 for simple debugging/demo
    )
    print("Dataset and DataLoader verified.")

    # 3. Model Initialization and Forward Pass
    print("\n--- Testing Model Architecture ---")
    model = IsotropicConvNeXtUNet(input_channels=8, num_classes=1).to(DEVICE)

    # Create dummy input batch (Batch Size=2, Channels=8, H=256, W=256)
    dummy_input = torch.randn(2, 8, 256, 256).to(DEVICE)

    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Expected output shape (2, 1, 256, 256), got {output.shape}"
    print("Model forward pass verified.")

    # 4. Loss Function Demonstration
    print("\n--- Testing Loss Function ---")
    criterion = HybridLoss()

    # Create dummy targets (binary mask)
    dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float().to(DEVICE)

    # Calculate loss
    loss = criterion(output, dummy_targets)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss value is NaN"
    print("Loss function verified.")

    # 5. Utility Functions Demonstration
    print("\n--- Testing Utility Functions ---")
    # Test RLE Encoding
    # Create a 3x3 mask with the center pixel set to 1
    # Flattened (column-major): 0, 0, 0, 0, 1, 0, 0, 0, 0
    # The '1' is at index 5 (1-based indexing)
    simple_mask = np.zeros((3, 3))
    simple_mask[1, 1] = 1
    rle_str = rle_encode(simple_mask)
    print(f"RLE for center pixel in 3x3: '{rle_str}'")
    assert rle_str == "5 1", f"Expected RLE '5 1', got '{rle_str}'"

    # Test Dice Coefficient
    dice_score = dice_coef(simple_mask, simple_mask)
    print(f"Dice Score (Perfect Match): {dice_score}")
    assert dice_score == 1.0, f"Expected Dice 1.0, got {dice_score}"
    print("Utilities verified.")

    # 6. Training Loop Demonstration
    print("\n--- Testing Training Loop (One Epoch) ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run training for one epoch on the small subset
    epoch_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
    )
    print(f"Epoch Loss: {epoch_loss:.4f}")
    assert isinstance(epoch_loss, float), "Epoch loss should be a float"
    print("Training loop verified.")

    # 7. Validation Loop Demonstration
    print("\n--- Testing Validation Loop ---")
    # Validate on the same small subset
    val_dice = validate(model, train_loader, DEVICE)
    print(f"Validation Global Dice: {val_dice:.4f}")
    assert 0.0 <= val_dice <= 1.0, "Dice score must be between 0 and 1"
    print("Validation loop verified.")

    # 8. Inference Demonstration
    print("\n--- Testing Inference ---")
    # Setup Test Dataset (using subset)
    test_dataset = ContrailDataset(
        split="test",
        transform=get_transforms("test"),
        max_samples=4,
        load_cached_data=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,  # Must be False for correct record_id mapping
        num_workers=0,
    )

    # Define output path for demo submission
    demo_submission_path = os.path.join("./working", "demo_submission.csv")

    # Run inference
    inference(model, test_loader, DEVICE, output_path=demo_submission_path)

    # Verify output
    assert os.path.exists(demo_submission_path), "Submission file was not created"

    df_sub = pd.read_csv(demo_submission_path)
    print(f"Submission file created with {len(df_sub)} rows.")
    print(f"Columns: {list(df_sub.columns)}")

    assert (
        len(df_sub) == 4
    ), "Submission should have 4 rows corresponding to the 4 test samples"
    assert (
        "record_id" in df_sub.columns and "encoded_pixels" in df_sub.columns
    ), "Missing required columns"
    print("Inference verified.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
