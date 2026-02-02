import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_factory import prepare_datasets, VentilatorDataset
from library.model import PCSDHNet
from library.trainer import Trainer


def generate_mock_data(num_breaths, file_path, is_test=False):
    """
    Generates a synthetic CSV file mimicking the ventilator dataset structure.
    Each breath consists of 80 time steps.
    """
    rows_per_breath = 80
    total_rows = num_breaths * rows_per_breath

    # Generate breath_ids
    breath_ids = np.repeat(np.arange(1, num_breaths + 1), rows_per_breath)

    # Generate time_steps (0 to ~3.0)
    time_steps = np.tile(np.linspace(0, 3.0, rows_per_breath), num_breaths)

    # Generate u_in (random control input)
    u_in = np.random.uniform(0, 100, total_rows)

    # Generate u_out (0 for first 30 steps, 1 for remaining 50 steps)
    u_out_pattern = np.concatenate([np.zeros(30), np.ones(50)])
    u_out = np.tile(u_out_pattern, num_breaths)

    # Generate R and C (random choice from typical values)
    r_vals = np.random.choice([5, 20, 50], num_breaths)
    c_vals = np.random.choice([10, 20, 50], num_breaths)
    R = np.repeat(r_vals, rows_per_breath)
    C = np.repeat(c_vals, rows_per_breath)

    # Generate ids
    ids = np.arange(1, total_rows + 1)

    data = {
        "id": ids,
        "breath_id": breath_ids,
        "R": R,
        "C": C,
        "time_step": time_steps,
        "u_in": u_in,
        "u_out": u_out.astype(int),
    }

    if not is_test:
        # Generate dummy pressure target
        # Simple physics proxy: Pressure ~ u_in * R + Volume / C
        # Just random noise for demo purposes is sufficient to check pipeline
        data["pressure"] = np.random.uniform(0, 50, total_rows)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df


def main():
    print("=== Starting PCSDH-Net Implementation Demo ===")

    # 1. Setup Environment and Overrides
    seed_everything(42)

    # Define demo directories
    demo_dir = "./working/demo_implementation"
    demo_input_dir = os.path.join(demo_dir, "input")
    demo_working_dir = os.path.join(demo_dir, "working")

    os.makedirs(demo_input_dir, exist_ok=True)
    os.makedirs(demo_working_dir, exist_ok=True)

    print(f"Created demo directories at {demo_dir}")

    # Override Config to use demo paths and settings for speed
    Config.TRAIN_PATH = os.path.join(demo_input_dir, "train.csv")
    Config.VAL_PATH = os.path.join(demo_input_dir, "validation.csv")
    Config.TEST_PATH = os.path.join(demo_input_dir, "test.csv")
    Config.SAMPLE_SUBMISSION_PATH = os.path.join(
        demo_input_dir, "sample_submission.csv"
    )
    Config.WORKING_DIR = demo_working_dir
    Config.OUTPUT_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce compute load for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.LSTM_LAYERS = 1  # Reduce complexity for speed
    Config.CNN_FILTERS = [32, 64]  # Reduce complexity for speed

    # 2. Generate Mock Data
    print("\n[Step 1] Generating Mock Data...")
    n_train_breaths = 20
    n_val_breaths = 10
    n_test_breaths = 10

    generate_mock_data(n_train_breaths, Config.TRAIN_PATH, is_test=False)
    generate_mock_data(n_val_breaths, Config.VAL_PATH, is_test=False)
    test_df = generate_mock_data(n_test_breaths, Config.TEST_PATH, is_test=True)

    # Create mock sample submission
    sample_sub = pd.DataFrame({"id": test_df["id"], "pressure": 0})
    sample_sub.to_csv(Config.SAMPLE_SUBMISSION_PATH, index=False)

    print("Mock data generated successfully.")

    # 3. Verify Data Factory
    print("\n[Step 2] Verifying Data Factory and Feature Engineering...")
    # Force reload to ignore any existing cache in the real working dir
    train_ds, val_ds, test_ds, scaler = prepare_datasets(load_cached_data=False)

    # Assertions for Dataset
    assert isinstance(train_ds, VentilatorDataset)
    assert (
        len(train_ds) == n_train_breaths
    ), f"Expected {n_train_breaths} breaths, got {len(train_ds)}"

    # Check item shape
    x_sample, y_sample, u_out_sample = train_ds[0]
    # Shape should be (80, n_features)
    # n_features is determined by Config.FEATURE_COLS (14 features)
    expected_features = len(Config.FEATURE_COLS)

    assert x_sample.shape == (
        80,
        expected_features,
    ), f"Expected shape (80, {expected_features}), got {x_sample.shape}"
    assert y_sample.shape == (80,), f"Expected target shape (80,), got {y_sample.shape}"
    assert u_out_sample.shape == (
        80,
    ), f"Expected u_out shape (80,), got {u_out_sample.shape}"

    print("Data Factory verification passed.")

    # 4. Verify Model Architecture
    print("\n[Step 3] Verifying Model Architecture...")
    model = PCSDHNet()
    model.eval()

    # Create dummy batch: (Batch, Seq, Features)
    dummy_input = torch.randn(2, 80, expected_features)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch, Seq, 1)
    assert output.shape == (
        2,
        80,
        1,
    ), f"Model output shape mismatch. Expected (2, 80, 1), got {output.shape}"
    print("Model architecture verification passed.")

    # 5. Verify Metric Logic
    print("\n[Step 4] Verifying Metric Calculation...")
    # Case 1: Perfect prediction
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([10, 20, 30, 40])
    u_out_in = np.array([0, 0, 1, 1])  # Only first two are inspiratory

    mae = compute_metric(y_pred, y_true, u_out_in)
    assert mae == 0.0, f"Expected MAE 0.0, got {mae}"

    # Case 2: Error only in expiratory phase (should be ignored)
    y_pred_exp_err = np.array([10, 20, 100, 100])
    mae_exp = compute_metric(y_pred_exp_err, y_true, u_out_in)
    assert mae_exp == 0.0, f"Expected MAE 0.0 (ignoring expiratory), got {mae_exp}"

    # Case 3: Error in inspiratory phase
    y_pred_insp_err = np.array([12, 18, 30, 40])  # Errors: |10-12|=2, |20-18|=2. Mean=2
    mae_insp = compute_metric(y_pred_insp_err, y_true, u_out_in)
    assert np.isclose(mae_insp, 2.0), f"Expected MAE 2.0, got {mae_insp}"

    print("Metric logic verification passed.")

    # 6. Verify Trainer (Training Loop)
    print("\n[Step 5] Verifying Training Loop...")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    trainer = Trainer()

    # Run training (1 epoch as configured)
    print("Running Trainer.fit()...")
    trainer.fit(train_loader, val_loader)

    # Check if best model was saved
    assert os.path.exists(trainer.best_model_path), "Best model file was not created."

    # Run inference
    print("Running Trainer.predict()...")
    preds = trainer.predict(test_loader)

    # Verify predictions shape
    expected_preds_len = n_test_breaths * 80
    assert (
        len(preds) == expected_preds_len
    ), f"Expected {expected_preds_len} predictions, got {len(preds)}"

    print("Training loop and inference verification passed.")

    # 7. Final Output Generation
    print("\n[Step 6] Generating Submission...")

    # We manually replicate the submission saving logic from trainer.py main() to verify it works with the output
    sample_sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    sample_sub_df["pressure"] = preds
    sample_sub_df.to_csv(Config.OUTPUT_PATH, index=False)

    assert os.path.exists(Config.OUTPUT_PATH), "Submission file not found."
    print(f"Submission saved to {Config.OUTPUT_PATH}")

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")


if __name__ == "__main__":
    main()
