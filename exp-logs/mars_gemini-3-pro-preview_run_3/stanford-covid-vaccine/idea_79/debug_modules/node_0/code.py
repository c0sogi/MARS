import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DeepResGLUBiGRU
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demo Execution ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment...")

    # Override Config for a fast demo run
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.DEBUG_SUBSET_SIZE = 100  # Train/Val on 100 samples only
    Config.WORKING_DIR = "./working/demo_run"

    # Update paths to use the new working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Set seed
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("[2/6] Loading data...")

    # Load data (force processing to ensure cache paths are valid for this run)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Data Shapes
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    targets = batch["targets"]

    print(
        f"Train Batch - Inputs: {inputs.shape}, Pairs: {pair_indices.shape}, Targets: {targets.shape}"
    )

    # Assertions
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, 107, 14)}, got {inputs.shape}"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Pair indices shape mismatch. Expected {(Config.BATCH_SIZE, 107)}, got {pair_indices.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 68, 5)}, got {targets.shape}"

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("[3/6] Initializing model...")

    model = DeepResGLUBiGRU()
    model.to(Config.DEVICE)

    # Dummy Forward Pass to verify architecture
    with torch.no_grad():
        dummy_in = inputs.to(Config.DEVICE)
        dummy_pairs = pair_indices.to(Config.DEVICE)
        dummy_out = model(dummy_in, dummy_pairs)

    print(f"Model Output Shape: {dummy_out.shape}")

    # Assert Output Shape (Batch, Seq_Len, Num_Targets)
    # Note: Model outputs predictions for full Seq_Len (107), even though targets are only 68.
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Model output mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {dummy_out.shape}"

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print("[4/6] Starting training loop...")

    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.fit()

    # Verify model checkpoint was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )
    print("Training finished and model saved.")

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("[5/6] Running inference on Test set...")

    # Load best model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch["inputs"].to(Config.DEVICE)
            pair_indices = batch["pair_indices"].to(Config.DEVICE)
            ids = batch["id"]

            preds = model(inputs, pair_indices)
            preds = preds.cpu().numpy()  # (Batch, 107, 5)

            all_preds.append(preds)
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"Total Predictions Shape: {all_preds.shape}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("[6/6] Generating submission file...")

    # Submission format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # We must flatten the predictions: (N_Samples, 107, 5) -> (N_Samples * 107, 6)

    submission_rows = []
    target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, sample_id in enumerate(all_ids):
        sample_pred = all_preds[idx]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_columns):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Submission Dimensions: {df_submission.shape}")

    # Verify Submission Logic
    # Test set has 240 samples. Each has 107 positions.
    # Total rows should be 240 * 107 = 25680.
    # Note: Even with DEBUG_SUBSET_SIZE, the library code provided does NOT slice the test set
    # (the line is commented out in get_dataloaders), so we expect full test set inference.
    expected_rows = 240 * 107
    assert (
        len(df_submission) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_submission)}"

    print("=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
