import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path so we can import from library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import RNAModel
from library.engine import train_fn, eval_fn


def run_demo():
    print("==== RNA Degradation Prediction Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # ---------------------------------------------------------
    # We modify the global Config to run a lightweight version of the task
    print("[1/6] Configuring lightweight parameters...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.HIDDEN_DIM = 64  # Reduced from 384 for speed
    Config.CONV_FILTERS = 32  # Reduced from 256
    Config.NUM_LAYERS = 2  # Logical depth
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Use a specific directory for this demo to avoid overwriting main experiment files
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"      Device: {device}")
    print(f"      Working Dir: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("[2/6] Loading and processing data...")
    # load_cached_data=False forces the Dataset to process from parquet files,
    # ensuring the preprocessing logic is exercised.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Data Integrity
    print("      Verifying batch structure...")
    batch = next(iter(train_loader))

    # Check required keys
    required_keys = {"features", "pair_indices", "pair_masks", "targets", "ids"}
    assert required_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Check Tensor Shapes
    # Features: (Batch, Seq_Len, Channels=14)
    assert batch["features"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    ), f"Feature shape mismatch: {batch['features'].shape}"

    # Targets: (Batch, Seq_Len, Targets=5)
    assert batch["targets"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch: {batch['targets'].shape}"

    print("      Batch verification passed.")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("[3/6] Initializing RNAModel...")
    model = RNAModel(Config).to(device)

    # Verify Forward Pass
    print("      Verifying forward pass...")
    with torch.no_grad():
        feats = batch["features"].to(device)
        p_idx = batch["pair_indices"].to(device)
        p_mask = batch["pair_masks"].to(device)

        output = model(feats, p_idx, p_mask)

    # Output should be (Batch, Seq_Len, 5)
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch: {output.shape}"

    print("      Forward pass verification passed.")

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("[4/6] Starting training loop (2 Epochs)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = torch.nn.MSELoss()

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, device, criterion)

        # Evaluate
        val_score = eval_fn(model, val_loader, device)

        print(
            f"      Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

        # Sanity checks
        assert not np.isnan(train_loss), "Training loss is NaN!"
        assert not np.isnan(val_score), "Validation score is NaN!"

    # Save the model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"      Model saved to {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 5. Inference
    # ---------------------------------------------------------
    print("[5/6] Running inference on Test set...")
    model.eval()

    # Dictionary to store predictions: id -> numpy array (107, 5)
    preds_store = {}

    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            p_idx = batch["pair_indices"].to(device)
            p_mask = batch["pair_masks"].to(device)
            ids = batch["ids"]

            outputs = model(feats, p_idx, p_mask)
            outputs_np = outputs.cpu().numpy()

            for i, sample_id in enumerate(ids):
                preds_store[sample_id] = outputs_np[i]

    print(f"      Generated predictions for {len(preds_store)} samples.")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("[6/6] Formatting submission file...")

    # Load sample submission to get the required ID_SeqPos structure
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Prepare lists for new columns
    # Target columns in order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    cols_data = {
        "reactivity": [],
        "deg_Mg_pH10": [],
        "deg_pH10": [],
        "deg_Mg_50C": [],
        "deg_50C": [],
    }

    # Iterate through the sample submission requirements
    # Format of id_seqpos: "id_00073f8be_0"
    for id_seqpos in sample_sub["id_seqpos"]:
        # Parse ID and Position
        parts = id_seqpos.split("_")
        # ID is everything except the last part (e.g., id_00073f8be)
        sample_id = "_".join(parts[:-1])
        seq_pos = int(parts[-1])

        # Retrieve prediction
        if sample_id in preds_store and seq_pos < Config.SEQ_LENGTH:
            pred_vector = preds_store[sample_id][seq_pos]
        else:
            # Fallback (should not happen for valid test IDs)
            pred_vector = np.zeros(Config.NUM_TARGETS)

        # Append to lists
        cols_data["reactivity"].append(pred_vector[0])
        cols_data["deg_Mg_pH10"].append(pred_vector[1])
        cols_data["deg_pH10"].append(pred_vector[2])
        cols_data["deg_Mg_50C"].append(pred_vector[3])
        cols_data["deg_50C"].append(pred_vector[4])

    # Create DataFrame
    submission_df = pd.DataFrame({"id_seqpos": sample_sub["id_seqpos"]})
    for col, data in cols_data.items():
        submission_df[col] = data

    # Verify shape
    assert (
        submission_df.shape == sample_sub.shape
    ), f"Submission shape mismatch. Expected {sample_sub.shape}, got {submission_df.shape}"

    # Save
    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"      Submission saved to {sub_path}")

    print("==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
