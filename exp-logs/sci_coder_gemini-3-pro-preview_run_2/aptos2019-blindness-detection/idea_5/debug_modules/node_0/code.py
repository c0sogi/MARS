import sys
import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_df, RetinopathyDataset, get_transforms, circle_crop
from library.modeling import RetinopathyModel, GeM
from library.engine import train_one_epoch, validate


def run_demonstration():
    # 1. Setup and Config Override for Speed
    print("=== Starting Demonstration ===")
    seed_everything(42)

    # Override CFG to ensure the script runs quickly (Debug Mode)
    CFG.debug = True
    CFG.debug_sample_size = 20  # Use only 20 samples
    CFG.epochs = 1
    CFG.batch_size = 4
    CFG.num_workers = 0  # Disable multiprocessing for simple script execution
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Configuration: Debug={CFG.debug}, Device={CFG.device}")

    # 2. Data Loading and Processing
    print("\n[Data] Loading DataFrames...")
    # Load metadata dataframes
    df_train = get_df("train")
    df_val = get_df("val")

    # Assertions to verify data loading logic
    assert (
        len(df_train) == CFG.debug_sample_size
    ), f"Expected {CFG.debug_sample_size} training samples, got {len(df_train)}"
    assert (
        len(df_val) == CFG.debug_sample_size
    ), f"Expected {CFG.debug_sample_size} validation samples, got {len(df_val)}"
    print(
        f"Loaded {len(df_train)} training samples and {len(df_val)} validation samples."
    )

    # Instantiate Datasets with transforms
    img_size = 256  # Use smaller size for faster demo
    train_dataset = RetinopathyDataset(
        df_train, transform=get_transforms("train", img_size)
    )
    val_dataset = RetinopathyDataset(df_val, transform=get_transforms("val", img_size))

    # Verify a single item from the dataset
    sample_img, sample_label = train_dataset[0]
    print(f"[Data] Sample Image Shape: {sample_img.shape}, Label: {sample_label}")

    # Check shapes: (Channels, Height, Width)
    assert sample_img.shape == (3, img_size, img_size), "Image tensor shape mismatch"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch tensor"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    # 3. Verify Utility Functions (Circle Crop)
    print("\n[Utils] Verifying Circle Crop...")
    # Create a synthetic image: 100x100 black with a 60x60 white square in the center
    synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)
    synthetic_img[20:80, 20:80, :] = 255

    cropped_img = circle_crop(synthetic_img)
    print(f"Original Size: {synthetic_img.shape}, Cropped Size: {cropped_img.shape}")

    # The crop should remove the black borders, resulting in a smaller image
    assert (
        cropped_img.shape[0] < 100 and cropped_img.shape[1] < 100
    ), "Circle crop did not reduce image dimensions"

    # 4. Model Instantiation
    print("\n[Model] Instantiating Model...")
    # We use 'resnet18' as a lightweight backbone for this demo instead of the heavy default
    model_name = "resnet18"
    model = RetinopathyModel(model_name=model_name, pretrained=False)
    model.to(CFG.device)

    # Verify GeM Pooling logic explicitly
    gem_layer = GeM(p=3)
    dummy_features = torch.randn(2, 64, 16, 16)  # Batch, Channels, H, W
    pooled_features = gem_layer(dummy_features)
    assert pooled_features.shape == (
        2,
        64,
        1,
        1,
    ), f"GeM output shape mismatch: {pooled_features.shape}"
    print(
        f"GeM Pooling check passed. Input: {dummy_features.shape} -> Output: {pooled_features.shape}"
    )

    # Verify full model forward pass
    dummy_input = torch.randn(CFG.batch_size, 3, img_size, img_size).to(CFG.device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Forward Pass Output Shape: {output.shape}")
    # Expecting (Batch_Size, 1) for regression
    assert output.shape == (CFG.batch_size, 1), "Model output shape mismatch"

    # 5. Training Loop
    print("\n[Engine] Running Training Epoch...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 1 epoch
    avg_train_loss = train_one_epoch(
        model, optimizer, train_loader, CFG.device, epoch=1
    )
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss returned NaN"

    # 6. Validation Loop
    print("\n[Engine] Running Validation...")
    avg_val_loss, preds, targets = validate(model, val_loader, CFG.device)

    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Predictions Shape: {preds.shape}, Targets Shape: {targets.shape}")

    assert len(preds) == len(
        val_dataset
    ), "Number of predictions does not match validation set size"
    assert len(targets) == len(
        val_dataset
    ), "Number of targets does not match validation set size"

    # 7. Metric Calculation
    print("\n[Metric] Calculating Quadratic Weighted Kappa...")
    # Calculate QWK on the validation results
    # (Note: Score might be low since model is untrained/trained briefly, checking logic only)
    qwk = quadratic_weighted_kappa(targets, preds)
    print(f"Validation QWK Score: {qwk:.4f}")

    # Verify metric bounds
    assert -1.0 <= qwk <= 1.0, "QWK score is outside valid range [-1, 1]"

    # Verify metric with synthetic perfect agreement
    perfect_score = quadratic_weighted_kappa(targets, targets)
    assert np.isclose(perfect_score, 1.0), "Metric should be 1.0 for perfect agreement"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
