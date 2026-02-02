import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_metric, get_device
from library.data import prepare_datasets, VentilatorDataset
from library.model import TAPINNet
from library.train import train_model

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data():
    """
    Creates a small subset of the data for demonstration purposes.
    This ensures the code runs quickly (seconds instead of hours).
    """
    print("Creating data subsets for demonstration...")

    # Define paths for demo data
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    # Read a small chunk of the real metadata
    # We need enough rows to get full breaths (80 steps per breath)
    # Reading 2000 rows should be plenty for ~20 breaths
    full_train = pd.read_csv(Config.TRAIN_CSV, nrows=80 * 50)
    full_test = pd.read_csv(Config.TEST_CSV, nrows=80 * 20)

    # Get unique breath IDs to ensure we don't split a breath in half
    train_breath_ids = full_train["breath_id"].unique()
    test_breath_ids = full_test["breath_id"].unique()

    # Select 10 breaths for train, 5 for val, 5 for test
    train_ids = train_breath_ids[:10]
    val_ids = train_breath_ids[10:15]
    test_ids = test_breath_ids[:5]

    # Filter dataframes
    demo_train = full_train[full_train["breath_id"].isin(train_ids)].copy()
    demo_val = full_train[full_train["breath_id"].isin(val_ids)].copy()
    demo_test = full_test[full_test["breath_id"].isin(test_ids)].copy()

    # Save to working directory
    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(f"  Train subset: {demo_train.shape}")
    print(f"  Val subset:   {demo_val.shape}")
    print(f"  Test subset:  {demo_test.shape}")

    return demo_train_path, demo_val_path, demo_test_path


def override_config(train_path, val_path, test_path):
    """
    Overrides the Config class attributes to use the demo datasets
    and temporary cache files.
    """
    print("Overriding Config for demo execution...")

    # Override Input Files
    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Override Cache Files (to avoid messing with real training artifacts)
    Config.TRAIN_CACHE_X = os.path.join(Config.WORKING_DIR, "demo_train_x.npy")
    Config.TRAIN_CACHE_Y = os.path.join(Config.WORKING_DIR, "demo_train_y.npy")
    Config.VAL_CACHE_X = os.path.join(Config.WORKING_DIR, "demo_val_x.npy")
    Config.VAL_CACHE_Y = os.path.join(Config.WORKING_DIR, "demo_val_y.npy")
    Config.TEST_CACHE_X = os.path.join(Config.WORKING_DIR, "demo_test_x.npy")
    Config.TEST_IDS = os.path.join(Config.WORKING_DIR, "demo_test_ids.npy")
    Config.SCALER_STATS = os.path.join(Config.WORKING_DIR, "demo_scaler_stats.npz")

    # Override Training Hyperparameters for Speed
    Config.BATCH_SIZE = 4  # Small batch size for small data
    Config.EPOCHS = 2  # Only 2 epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data


def verify_metric_logic():
    """
    Verifies the compute_metric function logic.
    Metric should only calculate MAE where u_out == 0 (Inspiratory phase).
    """
    print("\nVerifying Metric Logic...")

    # Case 1: Perfect prediction
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 5.0])
    u_out = torch.tensor([0, 0, 1, 1])  # Last two are expiratory (ignored)

    # Only first two should count. |1-1| + |2-2| = 0
    mae = compute_metric(preds, targets, u_out)
    assert mae == 0.0, f"Expected MAE 0.0, got {mae}"

    # Case 2: Error in inspiratory phase
    preds = torch.tensor([1.5, 2.0])
    targets = torch.tensor([1.0, 2.0])
    u_out = torch.tensor([0, 0])

    # |1.5-1.0| + |2.0-2.0| = 0.5 / 2 = 0.25
    mae = compute_metric(preds, targets, u_out)
    assert abs(mae - 0.25) < 1e-6, f"Expected MAE 0.25, got {mae}"

    print("  Metric logic verified.")


def verify_model_forward_pass():
    """
    Verifies that the TAPINNet model can be instantiated and process a batch.
    """
    print("\nVerifying Model Architecture...")

    device = get_device()
    model = TAPINNet().to(device)

    # Create dummy input: (Batch, Seq_Len, Features)
    # Seq_Len is fixed at 80 for this dataset
    batch_size = 2
    seq_len = 80
    features = Config.INPUT_DIM

    dummy_input = torch.randn(batch_size, seq_len, features).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check output shape: Should be (Batch, Seq_Len)
    expected_shape = (batch_size, seq_len)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"

    print(f"  Model forward pass successful. Output shape: {output.shape}")
    return model


def main():
    # 1. Setup
    seed_everything(42)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Create Data Subsets
    train_path, val_path, test_path = create_subset_data()

    # 3. Override Config
    override_config(train_path, val_path, test_path)

    # 4. Verify Data Pipeline
    print("\nRunning Data Pipeline (prepare_datasets)...")
    # Force reload to use the new subset files
    train_loader, val_loader, test_loader = prepare_datasets(load_cached_data=False)

    # Assertions on DataLoaders
    # We have 10 train breaths, batch size 4 -> 2 full batches (drop_last=True)
    assert len(train_loader) == 2, f"Expected 2 train batches, got {len(train_loader)}"

    # Check batch shape
    x, y = next(iter(train_loader))
    # x shape: (Batch, 80, Features)
    assert x.shape == (4, 80, Config.INPUT_DIM), f"Unexpected input shape: {x.shape}"
    # y shape: (Batch, 80)
    assert y.shape == (4, 80), f"Unexpected target shape: {y.shape}"

    print("  Data pipeline verified.")

    # 5. Verify Metric
    verify_metric_logic()

    # 6. Verify Model
    model = verify_model_forward_pass()

    # 7. Run Training
    print("\nRunning Training Loop...")
    # We use debug=True to ensure it doesn't run too many batches (though our data is small anyway)
    trained_model = train_model(
        epochs=Config.EPOCHS, load_cached_data=True, save_model=True, debug=True
    )

    # Verify model file exists
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Model checkpoint file was not created."
    print("  Training loop completed and model saved.")

    # 8. Inference on Test Set
    print("\nRunning Inference on Test Set...")
    trained_model.eval()
    predictions = []

    device = get_device()

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            preds = trained_model(x)
            predictions.append(preds.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)

    # Verify predictions match test IDs count
    # We loaded 5 test breaths * 80 steps = 400 steps
    test_ids = np.load(Config.TEST_IDS)
    assert len(all_preds) == len(
        test_ids
    ), f"Mismatch: {len(all_preds)} preds vs {len(test_ids)} IDs"

    # Create submission dataframe
    submission = pd.DataFrame({"id": test_ids, "pressure": all_preds})

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"  Inference successful. Submission saved to {submission_path}")
    print(f"  Submission head:\n{submission.head()}")


if __name__ == "__main__":
    main()
