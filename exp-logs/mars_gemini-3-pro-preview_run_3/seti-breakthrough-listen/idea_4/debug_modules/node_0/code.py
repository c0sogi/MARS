import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import CadenceDataset
from library.model import SiameseEfficientNet
from library.engine import train_one_epoch, validate, generate_submission


def create_mini_metadata(source_path, dest_path, n_samples=16):
    """
    Creates a small subset of the metadata CSV for rapid demonstration.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample top n_samples to ensure we have valid files
    df_subset = df.head(n_samples).copy()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df_subset.to_csv(dest_path, index=False)
    print(f"Created mini metadata at {dest_path} with {len(df_subset)} samples.")
    return dest_path


def main():
    # 1. Setup
    print("--- Setting up environment ---")
    seed_everything(Config.SEED)

    # Define paths for the demo
    demo_dir = "./working/demo"
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_path = os.path.join(demo_dir, "train_mini.csv")
    mini_val_path = os.path.join(demo_dir, "val_mini.csv")
    mini_test_path = os.path.join(demo_dir, "test_mini.csv")

    # 2. Prepare Data Subsets
    print("--- Preparing data subsets ---")
    create_mini_metadata(Config.TRAIN_CSV, mini_train_path, n_samples=16)
    create_mini_metadata(Config.VAL_CSV, mini_val_path, n_samples=8)
    create_mini_metadata(Config.TEST_CSV, mini_test_path, n_samples=8)

    # 3. Verify Dataset Logic
    print("--- Verifying Dataset Logic ---")
    # Initialize dataset with the mini training file
    train_ds = CadenceDataset(metadata_path=mini_train_path, mode="train")

    # Check length
    assert len(train_ds) == 16, f"Expected 16 samples, got {len(train_ds)}"

    # Check item structure
    sample = train_ds[0]
    required_keys = ["on_input", "off_input", "target", "id"]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset output"

    # Check shapes
    # Expected shape after transform: (3, 224, 224)
    expected_shape = (3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        sample["on_input"].shape == expected_shape
    ), f"Incorrect on_input shape: {sample['on_input'].shape}"
    assert (
        sample["off_input"].shape == expected_shape
    ), f"Incorrect off_input shape: {sample['off_input'].shape}"
    assert isinstance(sample["target"], torch.Tensor), "Target should be a tensor"

    print("Dataset verification passed.")

    # 4. Verify Model Logic
    print("--- Verifying Model Logic ---")
    device = Config.DEVICE
    # Use pretrained=False for speed/offline capability during demo
    model = SiameseEfficientNet(pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy batch
    dummy_on = torch.randn(2, 3, 224, 224).to(device)
    dummy_off = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_on, dummy_off)

    # Check output shape (Batch, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model forward pass verification passed.")

    # 5. Verify Training Loop (Engine)
    print("--- Verifying Training Engine ---")
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple debug to avoid multiprocessing overhead
        pin_memory=True,
    )

    val_ds = CadenceDataset(metadata_path=mini_val_path, mode="val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Run one epoch of training
    print("Running training step...")
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch=1)

    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print(f"Training step completed. Loss: {train_loss:.4f}")

    # Run validation
    print("Running validation step...")
    val_loss, val_auc = validate(model, val_loader, device)

    assert isinstance(val_loss, float), "Val loss should be a float"
    assert isinstance(val_auc, float), "Val AUC should be a float"
    print(f"Validation step completed. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 6. Verify Submission Generation
    print("--- Verifying Submission Generation ---")
    test_ds = CadenceDataset(metadata_path=mini_test_path, mode="test")
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    submission_path = os.path.join(demo_dir, "submission_demo.csv")

    generate_submission(model, test_loader, device, submission_path)

    # Check if file exists
    assert os.path.exists(submission_path), "Submission file was not created"

    # Check content
    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == 8, f"Expected 8 predictions, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Missing columns in submission"
    assert df_sub["target"].dtype == float, "Target column should be float"

    print(f"Submission verification passed. File saved to {submission_path}")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
