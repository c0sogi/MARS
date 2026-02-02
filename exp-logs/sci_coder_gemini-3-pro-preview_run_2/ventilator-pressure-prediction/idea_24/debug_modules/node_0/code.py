import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.data_processing import load_dataset, get_ventilator_dataset
from library.dataset import VentilatorDataset
from library.model import DeepBiLSTM
from library.loss import WeightedL1Loss, competition_metric
from library.trainer import Trainer


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    # We modify the Config class attributes directly to run a fast debug session
    print("Overriding Config for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 breaths
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.CACHE_DIR = "./working/demo_cache"  # Use a separate cache dir
    Config.SUBMISSION_FILE = "./working/demo_submission.csv"

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    set_seed(Config.SEED)

    # ==========================================
    # 2. Data Processing & Dataset Verification
    # ==========================================
    print("\n--- Verifying Data Processing ---")

    # Load train data using the library function (Debug mode)
    # This triggers feature engineering, scaling, and reshaping
    X_train, y_train, u_out_train = load_dataset(
        "train", debug=True, load_cached_data=False
    )

    # Verify Shapes
    # Expected: (Num_Breaths, 80, Input_Dim)
    print(f"Train X Shape: {X_train.shape}")
    print(f"Train y Shape: {y_train.shape}")

    assert len(X_train.shape) == 3, "X should be 3D (Batch, Seq, Feat)"
    assert X_train.shape[1] == 80, "Sequence length must be 80"
    assert (
        X_train.shape[0] == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} breaths"
    assert y_train.shape == (Config.DEBUG_SAMPLE_SIZE, 80), "Target shape mismatch"
    assert u_out_train.shape == (Config.DEBUG_SAMPLE_SIZE, 80), "u_out shape mismatch"

    # Instantiate Dataset
    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)

    # Verify __getitem__
    sample_x, sample_y, sample_u = train_dataset[0]
    assert torch.is_tensor(sample_x), "Dataset should return tensors"
    assert sample_x.dtype == torch.float32, "Features should be float32"
    assert sample_x.shape[0] == 80, "Sample sequence length mismatch"

    print("Data Processing and Dataset verified successfully.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    device = torch.device("cpu")  # Use CPU for simple logic verification
    model = DeepBiLSTM(
        input_dim=Config.INPUT_DIM,
        hidden_size=64,  # Smaller hidden size for demo
        num_layers=2,
        glu_width=32,
    ).to(device)

    # Create a dummy batch
    dummy_input = torch.randn(4, 80, Config.INPUT_DIM).to(device)

    # Forward pass
    output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Verify Output
    # Expected: (Batch, Seq_Len) -> (4, 80)
    assert output.shape == (4, 80), f"Expected output shape (4, 80), got {output.shape}"

    print("Model architecture verified successfully.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n--- Verifying Loss Function ---")

    criterion = WeightedL1Loss()

    # Construct a deterministic scenario
    # Preds: [10, 10], Targets: [12, 12] -> Abs Error: [2, 2]
    # u_out: [0, 1] -> 0 is Inspiratory (Weight 1.0), 1 is Expiratory (Weight 0.1)

    preds = torch.tensor([[10.0, 10.0]])
    targets = torch.tensor([[12.0, 12.0]])
    u_out = torch.tensor([[0.0, 1.0]])

    # Expected Calculation:
    # Item 1 (Insp): |10-12| * 1.0 = 2.0
    # Item 2 (Exp):  |10-12| * 0.1 = 0.2
    # Mean: (2.0 + 0.2) / 2 = 1.1

    loss_val = criterion(preds, targets, u_out)
    print(f"Calculated Loss: {loss_val.item()}")

    assert (
        abs(loss_val.item() - 1.1) < 1e-6
    ), f"Loss calculation incorrect. Expected 1.1, got {loss_val.item()}"

    # Verify Competition Metric (MAE on Inspiratory Phase Only)
    # Should only consider the first item (u_out=0), error is 2.0
    metric_val = competition_metric(preds, targets, u_out)
    print(f"Competition Metric: {metric_val}")

    assert (
        abs(metric_val - 2.0) < 1e-6
    ), f"Metric calculation incorrect. Expected 2.0, got {metric_val}"

    print("Loss functions verified successfully.")

    # ==========================================
    # 5. Training Loop Verification
    # ==========================================
    print("\n--- Verifying Training & Inference Loop ---")

    # Initialize Trainer
    # Note: Trainer initializes its own model using Config parameters.
    # Since we updated Config earlier, it will use the debug settings.
    trainer = Trainer()

    # Prepare DataLoaders
    # We need val and test sets as well
    print("Loading Val and Test sets (Debug)...")
    X_val, y_val, u_val = load_dataset("val", debug=True, load_cached_data=False)
    val_dataset = VentilatorDataset(X_val, y_val, u_val)

    X_test, _, u_test = load_dataset("test", debug=True, load_cached_data=False)
    test_dataset = VentilatorDataset(X_test, None, u_test)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print("Starting Training Fit...")
    # This runs the training loop for Config.EPOCHS (2)
    trainer.fit(train_loader, val_loader)

    # Check if best model was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print("Training completed. Best model saved.")

    print("Starting Inference...")
    # This generates predictions and saves submission.csv
    trainer.predict(test_loader)

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")
    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission columns missing"

    # Verify row count matches test metadata for the debug subset
    # Note: The Trainer loads the FULL test metadata to align IDs.
    # However, our test_loader only predicted on the DEBUG subset (50 breaths * 80 steps = 4000 rows).
    # The Trainer.predict method in the provided code concatenates predictions and then aligns with metadata.
    # If the provided Trainer code assumes full test set, it might mismatch if we only feed a partial loader.
    # Let's check the Trainer code logic provided in the prompt.
    # Trainer.predict: "final_predictions = np.concatenate(all_preds)" -> "df_test_meta = pd.read_csv(Config.TEST_METADATA)"
    # It loads the FULL metadata. If we only predict on a subset, len(final_predictions) != len(df_test_meta).
    # This would raise a ValueError in the real library code.
    # HOWEVER, in this specific debug run, we must ensure we don't crash.
    # Since we cannot modify library code, we must acknowledge that `trainer.predict` might fail if
    # the loader size doesn't match metadata size.
    # BUT: The `load_dataset` function with `debug=True` subsets the metadata internally if we were using it for loading.
    # The `Trainer.predict` function reads `Config.TEST_METADATA` directly from disk.
    # To make this pass without erroring on the library's size check, we should probably
    # temporarily overwrite the TEST_METADATA file with a subsetted version for this demo,
    # or accept that we've demonstrated up to the point of failure.
    #
    # BETTER APPROACH: We can create a temporary metadata file for the debug subset and point Config to it.

    print("Training and Inference loop logic executed.")


if __name__ == "__main__":
    # We need to handle the metadata mismatch for the prediction step mentioned above
    # to ensure the script finishes successfully.

    # 1. Load original test metadata
    full_test_meta = pd.read_csv(Config.TEST_METADATA)

    # 2. Subset to match the debug sample size (50 breaths)
    unique_breaths = full_test_meta[Config.BREATH_ID_COL].unique()[:50]
    subset_meta = full_test_meta[
        full_test_meta[Config.BREATH_ID_COL].isin(unique_breaths)
    ]

    # 3. Save to a temp location
    temp_meta_path = "./working/temp_test_metadata.csv"
    subset_meta.to_csv(temp_meta_path, index=False)

    # 4. Point Config to this temp file so Trainer.predict loads the correct matching metadata
    Config.TEST_METADATA = temp_meta_path

    # Run main
    main()

    print("\nDemonstration complete.")
