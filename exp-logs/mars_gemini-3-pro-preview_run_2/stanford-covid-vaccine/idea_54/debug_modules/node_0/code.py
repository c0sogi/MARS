import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.data import get_loader
from library.model import DARDN
from library.loss import MCRMSELoss
from library.train import train_model, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== RNA Degradation Prediction Demo ===\n")

    # 1. Setup and Configuration Override for Speed
    # We override the Config class attributes directly to ensure the demo runs quickly.
    print("[1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 100  # Use a small subset
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set reproducible seed
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.EPOCHS}")

    # 2. Data Loading and Verification
    print("\n[2] Loading Data and Verifying Shapes...")
    train_loader = get_loader("train", batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fetch one batch to verify
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    partner_indices = batch["partner_indices"].to(device)
    targets = batch["targets"].to(device)

    print(f"    Input shape: {inputs.shape} (Expected: [B, 107, 18])")
    print(f"    Partner indices shape: {partner_indices.shape} (Expected: [B, 107])")
    print(f"    Targets shape: {targets.shape} (Expected: [B, 107, 5])")

    # Assertions
    assert (
        inputs.shape[1] == Config.SEQ_LENGTH
    ), f"Sequence length mismatch: {inputs.shape[1]}"
    assert inputs.shape[2] == 18, f"Feature dimension mismatch: {inputs.shape[2]}"
    assert (
        targets.shape[2] == Config.NUM_TARGETS
    ), f"Target dimension mismatch: {targets.shape[2]}"

    # 3. Model Instantiation and Forward Pass
    print("\n[3] Instantiating Model and Running Forward Pass...")
    model = DARDN().to(device)

    # The model returns two outputs: y_1 (first pass) and y_2 (refined pass)
    y_1, y_2 = model(inputs, partner_indices)

    print(f"    Output y_1 shape: {y_1.shape}")
    print(f"    Output y_2 shape: {y_2.shape}")

    assert y_1.shape == targets.shape, "y_1 shape mismatch"
    assert y_2.shape == targets.shape, "y_2 shape mismatch"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Calculation...")
    criterion = MCRMSELoss().to(device)
    loss = criterion(y_2, targets)

    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # 5. Training Loop Execution
    print("\n[5] Executing Training Loop...")
    # train_model handles the loop, validation, and saving best_model.pth
    best_score = train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    print(f"    Training finished. Best Validation Score: {best_score:.4f}")
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."

    # 6. Inference and Submission Generation
    print("\n[6] Running Inference on Test Set...")

    # Load test data
    test_loader = get_loader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load best model weights
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    preds_list = []
    ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            t_inputs = batch["inputs"].to(device)
            t_partners = batch["partner_indices"].to(device)
            t_ids = batch["ids"]

            # Forward pass - use refined output y_2
            _, y_pred = model(t_inputs, t_partners)

            # Move to CPU
            y_pred = y_pred.cpu().numpy()

            preds_list.append(y_pred)
            ids_list.extend(t_ids)

    # Concatenate all predictions: (Total_Samples, Seq_Len, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    print(f"    Total Test Predictions shape: {all_preds.shape}")

    # 7. Formatting Submission
    print("\n[7] Formatting Submission File...")

    submission_data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seq_pos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_preds = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[col_idx]

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Verify submission format
    expected_cols = ["id_seqpos"] + target_cols
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check row count: num_test_samples * seq_length
    # Note: In DEBUG mode, we only loaded MAX_DEBUG_SAMPLES for test as well
    expected_rows = len(ids_list) * Config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Got {len(df_sub)}, expected {expected_rows}"

    # Save submission
    output_path = "demo_submission.csv"
    df_sub.to_csv(output_path, index=False)
    print(f"    Submission saved to {output_path}")
    print(f"    First 5 rows:\n{df_sub.head().to_string()}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
