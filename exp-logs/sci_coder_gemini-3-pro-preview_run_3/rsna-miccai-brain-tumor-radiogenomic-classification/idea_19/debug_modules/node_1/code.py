import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import BraTSDataset
from library.network import SHDNet
from library.engine import train_model, predict


def run_demo():
    # 1. Setup
    print(">>> Setting up demo environment...")
    seed_everything(42)
    device = get_device()

    # Define paths for demo
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_meta = os.path.join(demo_dir, "metadata", "train.parquet")
    demo_val_meta = os.path.join(demo_dir, "metadata", "val.parquet")
    demo_test_meta = os.path.join(demo_dir, "metadata", "test.parquet")
    os.makedirs(os.path.dirname(demo_train_meta), exist_ok=True)

    # Override Config cache dir to separate demo cache from main cache
    # We do this by patching the class attribute for the scope of this run
    Config.CACHE_DIR = demo_dir

    # 2. Prepare Data Subsets (Optimization for Speed)
    print(">>> Creating metadata subsets for speed...")

    # Load original metadata
    df_train_full = pd.read_parquet(Config.TRAIN_META_PATH)
    df_val_full = pd.read_parquet(Config.VAL_META_PATH)
    df_test_full = pd.read_parquet(Config.TEST_META_PATH)

    # Create subsets (8 train, 4 val, 4 test)
    # This ensures the code runs quickly while still processing real DICOMs
    df_train_sub = df_train_full.head(8)
    df_val_sub = df_val_full.head(4)
    df_test_sub = df_test_full.head(4)

    # Save subsets
    df_train_sub.to_parquet(demo_train_meta, index=False)
    df_val_sub.to_parquet(demo_val_meta, index=False)
    df_test_sub.to_parquet(demo_test_meta, index=False)

    print(
        f"Subset sizes - Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )

    # 3. Verify Dataset Class
    print(">>> Verifying BraTSDataset...")
    # Initialize dataset with the subset metadata
    # cache_name ensures we create new cache files for this demo
    train_ds = BraTSDataset(
        metadata_path=demo_train_meta, cache_name="cached_train", is_train=True
    )

    # Assertions
    assert len(train_ds) == 8, f"Expected 8 samples, got {len(train_ds)}"

    # Check item shape
    sample_x, sample_y = train_ds[0]
    # Expected shape: (128, 256, 256) -> 4 modalities * 32 slices
    assert sample_x.shape == (
        128,
        256,
        256,
    ), f"Unexpected input shape: {sample_x.shape}"
    assert isinstance(sample_y, torch.Tensor), "Target should be a tensor"

    print("Dataset verification passed.")

    # 4. Verify Model Class
    print(">>> Verifying SHDNet...")
    model = SHDNet(drop_path_rate=0.1).to(device)

    # Create dummy input batch (Batch=2, Channels=128, H=256, W=256)
    dummy_input = torch.randn(2, 128, 256, 256).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model verification passed.")

    # 5. Run Training Loop
    print(">>> Running Training Loop (Demo)...")

    # Setup Dataloaders
    val_ds = BraTSDataset(
        metadata_path=demo_val_meta, cache_name="cached_val", is_train=True
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Setup Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    save_path = os.path.join(demo_dir, "best_model.pth")

    # Train for 2 epochs
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        save_path=save_path,
        patience=2,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print("Training loop completed successfully.")

    # 6. Run Inference
    print(">>> Running Inference (Demo)...")

    # Setup Test Dataset
    test_ds = BraTSDataset(
        metadata_path=demo_test_meta, cache_name="cached_test", is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Predict
    ids, probs = predict(model, test_loader, device)

    # Assertions
    assert len(ids) == 4, "Number of predictions does not match test set size"
    assert len(probs) == 4, "Number of probabilities does not match test set size"
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of range [0, 1]"

    # Save demo submission
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": probs})
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Inference completed. Submission saved to {submission_path}")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
