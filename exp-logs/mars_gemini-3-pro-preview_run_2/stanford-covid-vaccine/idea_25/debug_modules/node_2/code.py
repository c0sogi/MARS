import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.data import get_dataloaders
from library.model import StackedInteractionDenseNet
from library.train import masked_mcrmse_loss, train_model

# --------------------------------------------------------------------------
# 1. Configuration Setup for Demo
# --------------------------------------------------------------------------
# Override Config attributes to ensure the demo runs quickly and efficiently
Config.DEBUG = True
Config.DEBUG_SAMPLES = 40  # Use a tiny subset of data
Config.EPOCHS = 2  # Run only 2 epochs
Config.BATCH_SIZE = 4  # Small batch size
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

# Set up a specific working directory for this demo
Config.WORKING_DIR = "./working/demo_execution"
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Redirect cache and model paths to the demo directory
Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_demo.npz")
Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_demo.npz")
Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_demo.npz")
Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

if __name__ == "__main__":
    print("Starting Demo Script...")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Data Loading ---")
    # get_dataloaders uses Config.DEBUG to slice the dataset
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    partners = batch["partner_indices"]
    targets = batch["targets"]
    mask = batch["mask"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Inputs shape: {inputs.shape}")

    # Assertions to verify data shapes
    # Inputs: (Batch, Seq_Len, Input_Dim=19)
    assert inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.INPUT_DIM)
    # Partners: (Batch, Seq_Len)
    assert partners.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    # Targets: (Batch, Seq_Len, Num_Targets=5)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    # Mask: (Batch, Seq_Len)
    assert mask.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)

    print("Data shapes verified.")

    # --------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # --------------------------------------------------------------------------
    print("\n--- Model Verification ---")
    model = StackedInteractionDenseNet().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partners = partners.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    # Perform forward pass
    outputs = model(inputs, partners)
    print(f"Model Output shape: {outputs.shape}")

    # Verify output shape and content
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"
    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n--- Loss Function Verification ---")
    loss = masked_mcrmse_loss(outputs, targets, mask)
    print(f"Calculated Loss: {loss.item():.6f}")

    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"

    # Verify MetricTracker
    tracker = MetricTracker()
    tracker.update(outputs, targets, mask)
    score = tracker.compute()
    print(f"MetricTracker Score: {score:.6f}")
    assert score >= 0
    print("Loss and Metric verification successful.")

    # --------------------------------------------------------------------------
    # 5. Full Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n--- Executing Training Loop (Demo) ---")
    # train_model uses the Config class we modified earlier
    train_model()

    # Verify that the model was saved
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print(f"Training loop completed. Model saved to {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Inference and Submission ---")
    # Load the best model saved during training
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    preds_list = []
    ids_list = []

    # Run inference on test set
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partners = batch["partner_indices"].to(device)
            ids = batch["id"]

            outputs = model(inputs, partners)
            outputs = outputs.cpu().numpy()

            preds_list.append(outputs)
            ids_list.extend(ids)

    preds_arr = np.concatenate(preds_list, axis=0)
    print(f"Inference array shape: {preds_arr.shape}")

    # Generate Submission CSV
    # Format requires a row for every sequence position: id_seqpos
    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_arr[i]  # Shape: (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[col_idx]

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission dataframe shape: {df_sub.shape}")

    # Verify submission integrity
    # Columns: id_seqpos + 5 targets = 6 columns
    assert df_sub.shape[1] == 6
    # Rows: Number of test samples * Sequence Length
    expected_rows = len(ids_list) * Config.SEQ_LENGTH
    assert len(df_sub) == expected_rows

    print("\nDemo completed successfully.")
