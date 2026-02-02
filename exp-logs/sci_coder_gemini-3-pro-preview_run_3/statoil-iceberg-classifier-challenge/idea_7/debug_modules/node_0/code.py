import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import (
    set_seed,
    DEVICE,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
)
from library.dataset import process_and_cache_data, IcebergDataset, get_transforms
from library.model import HMP_CNN
from library.engine import train_one_epoch, evaluate, predict_with_tta, save_submission
from library.utils import save_checkpoint, load_checkpoint


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Set Seed for Reproducibility
    set_seed(42)
    print("Random seed set.")

    # 2. Data Processing and Caching
    # This function loads the raw JSONs, converts bands to numpy arrays, and saves .npy files.
    # It handles the 3-channel creation (HH, HV, Avg) internally.
    print("\n--- Processing and Caching Data ---")

    # Process training data
    full_train_data = process_and_cache_data("train", load_cached_data=True)
    assert "X" in full_train_data
    assert "angle" in full_train_data
    assert "y" in full_train_data
    assert full_train_data["X"].shape[1:] == (75, 75, 3)
    print("Training data processed successfully.")

    # Process test data
    # Note: Loading test.json might take a moment, but it's required to get the arrays.
    full_test_data = process_and_cache_data("test", load_cached_data=True)
    assert "X" in full_test_data
    assert "ids" in full_test_data
    print("Test data processed successfully.")

    # 3. Dataset Instantiation (Using Subsets for Speed)
    print("\n--- Initializing Datasets (Subset) ---")

    # Load metadata
    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    df_val_meta = pd.read_csv(VAL_META_PATH)
    df_test_meta = pd.read_csv(TEST_META_PATH)

    # Use a small subset for demonstration to ensure speed
    subset_size = 32
    df_train_subset = df_train_meta.head(subset_size).copy()
    df_val_subset = df_val_meta.head(subset_size).copy()
    df_test_subset = df_test_meta.head(subset_size).copy()

    # Calculate mean angle from training data for imputation
    angle_mean = np.nanmean(full_train_data["angle"])

    # Create Train Dataset
    train_dataset = IcebergDataset(
        metadata_df=df_train_subset,
        full_data_dict=full_train_data,
        transform=get_transforms("train"),
        angle_fill_value=angle_mean,
        mode="train",
    )

    # Create Validation Dataset
    val_dataset = IcebergDataset(
        metadata_df=df_val_subset,
        full_data_dict=full_train_data,
        transform=get_transforms("val"),
        angle_fill_value=angle_mean,
        mode="val",
    )

    # Verify item structure
    img, angle, target = train_dataset[0]
    # Image should be (3, 75, 75) tensor
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 75, 75)
    # Angle should be (1,) tensor
    assert angle.shape == (1,)
    # Target should be (1,) tensor
    assert target.shape == (1,)
    print(f"Dataset verification passed. Image shape: {img.shape}")

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    model = HMP_CNN()
    model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}.")

    # Verify Forward Pass
    dummy_img = torch.randn(2, 3, 75, 75).to(DEVICE)
    dummy_angle = torch.tensor([[35.0], [40.0]]).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    assert output.shape == (2, 1)
    assert 0.0 <= output.min() and output.max() <= 1.0
    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n--- Running Training Loop (1 Epoch) ---")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Train for one epoch
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device=DEVICE)
    print(f"Epoch finished. Average Training Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss)

    # 6. Evaluation Demonstration
    print("\n--- Running Evaluation ---")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    val_loss = evaluate(model, val_loader, criterion, device=DEVICE)
    print(f"Evaluation finished. Average Validation Loss: {val_loss:.4f}")

    # 7. Checkpointing
    print("\n--- Testing Checkpoint Saving/Loading ---")
    checkpoint_dir = os.path.join(WORKING_DIR, "demo_checkpoints")

    # Save
    save_checkpoint(
        state={
            "epoch": 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": val_loss,
        },
        is_best=True,
        checkpoint_dir=checkpoint_dir,
    )
    assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pth"))
    assert os.path.exists(os.path.join(checkpoint_dir, "model_best.pth"))

    # Load
    new_model = HMP_CNN().to(DEVICE)
    start_epoch, best_score = load_checkpoint(
        os.path.join(checkpoint_dir, "model_best.pth"), new_model, device=DEVICE
    )
    assert start_epoch == 1
    assert abs(best_score - val_loss) < 1e-6
    print("Checkpoint save/load verified.")

    # 8. Inference with TTA
    print("\n--- Running Inference with TTA ---")
    test_dataset = IcebergDataset(
        metadata_df=df_test_subset,
        full_data_dict=full_test_data,
        transform=get_transforms("test"),
        angle_fill_value=angle_mean,
        mode="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Predict
    predictions = predict_with_tta(model, test_loader, device=DEVICE)

    # Verify predictions
    assert len(predictions) == len(df_test_subset)
    first_id = df_test_subset.iloc[0]["id"]
    assert first_id in predictions
    assert 0.0 <= predictions[first_id] <= 1.0
    print(f"Generated {len(predictions)} predictions.")

    # 9. Submission Generation
    print("\n--- Generating Submission File ---")
    demo_submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    save_submission(predictions, output_path=demo_submission_path)

    assert os.path.exists(demo_submission_path)
    df_sub = pd.read_csv(demo_submission_path)
    assert list(df_sub.columns) == ["id", "is_iceberg"]
    assert len(df_sub) == len(df_test_subset)
    print(f"Submission file saved to {demo_submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
