import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.dataset import get_data, RNADataset
from library.model import RNA_Net
from library.loss import DeepSupervisionLoss
from library.train import Trainer


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring Environment for Demo...")

    # Override Config for speed and demonstration purposes
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.HIDDEN_DIM = 64  # Reduced from 384
    Config.NUM_LAYERS = 2  # Reduced from 6
    Config.EMBED_DIM_SEQ = 32  # Reduced
    Config.EMBED_DIM_LOOP = 16  # Reduced
    Config.EMBED_DIM_DIST = 16  # Reduced
    # Update total embed dim derived from others
    Config.EMBED_DIM = (
        Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST
    )

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Setup environment (creates dirs, sets seeds)
    device = Config.setup_environment(seed=42)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("\n[2] Loading and Verifying Data...")

    # Load data using the library function
    # This will read from ./metadata/*.parquet and cache to ./working/demo_run/*.npz
    full_train_data = get_data(mode="train", load_cached_data=False)
    full_val_data = get_data(mode="val", load_cached_data=False)

    # Create a mini-dataset to ensure the training loop finishes quickly
    MINI_SIZE = 32
    print(f"    Creating mini-datasets of size {MINI_SIZE}...")

    def create_mini_subset(data_dict, size):
        return {k: v[:size] for k, v in data_dict.items() if isinstance(v, np.ndarray)}

    train_data = create_mini_subset(full_train_data, MINI_SIZE)
    val_data = create_mini_subset(full_val_data, MINI_SIZE)

    # Verify Data Shapes
    # Sequences: (N, 107)
    assert train_data["sequences"].shape == (
        MINI_SIZE,
        Config.SEQ_LEN,
    ), "Incorrect sequence shape"
    # Targets: (N, 68, 3) -> 3 target columns
    assert train_data["targets"].shape == (
        MINI_SIZE,
        Config.PRED_LEN,
        3,
    ), "Incorrect target shape"
    # Distances: (N, 107, Embed_Dim_Dist)
    assert train_data["distances"].shape == (
        MINI_SIZE,
        Config.SEQ_LEN,
        Config.EMBED_DIM_DIST,
    ), "Incorrect distance shape"

    print("    Data shapes verified successfully.")

    # Instantiate Datasets and Loaders
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 3. Model & Loss Logic Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture and Loss Logic...")

    model = RNA_Net().to(device)
    criterion = DeepSupervisionLoss()

    # Create dummy batch
    dummy_seq = torch.randint(0, 4, (2, Config.SEQ_LEN)).to(device)
    dummy_loop = torch.randint(0, 7, (2, Config.SEQ_LEN)).to(device)
    dummy_dist = torch.randn(2, Config.SEQ_LEN, Config.EMBED_DIM_DIST).to(device)
    dummy_target = torch.randn(2, Config.PRED_LEN, 3).to(device)

    # Forward Pass
    main_pred, layer_preds = model(dummy_seq, dummy_loop, dummy_dist)

    # Check Output Shapes
    # Main prediction: (Batch, Seq_Len, 3) - Note: Model outputs full seq len, loss slices it
    assert main_pred.shape == (
        2,
        Config.SEQ_LEN,
        3,
    ), f"Expected (2, 107, 3), got {main_pred.shape}"
    # Deep supervision: Should have 1 (Stem) + NUM_LAYERS predictions
    expected_layers = 1 + Config.NUM_LAYERS
    assert (
        len(layer_preds) == expected_layers
    ), f"Expected {expected_layers} layer preds, got {len(layer_preds)}"

    # Loss Calculation
    loss = criterion(main_pred, layer_preds, dummy_target)

    # Verify Loss
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    print("    Model forward pass and loss calculation verified.")

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[4] Executing Training Loop (Mini-Batch)...")

    trainer = Trainer(device=device)
    # Overwrite the internal model with our verified/configured one (though Trainer init does this too)
    # We need to ensure the Trainer uses the Config values we set globally.
    # Since Trainer instantiates RNA_Net inside __init__, and RNA_Net uses Config.*,
    # the trainer.model already respects our reduced dimensions.

    # Run training
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=2)

    # Verify Checkpoint
    if os.path.exists(Config.MODEL_PATH):
        print(f"    Checkpoint successfully saved at: {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # ---------------------------------------------------------
    # 5. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n[5] Simulating Inference and Submission Generation...")

    # Load Test Data (Mini subset)
    full_test_data = get_data(mode="test", load_cached_data=False)
    test_data = create_mini_subset(full_test_data, MINI_SIZE)
    test_dataset = RNADataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load Best Model
    best_model = RNA_Net().to(device)
    best_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    best_model.eval()

    # Inference Loop
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for seq, loop, dist, _ in test_loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward
            preds, _ = best_model(seq, loop, dist)

            # Move to CPU
            preds = preds.cpu().numpy()  # (B, 107, 3)

            all_preds.append(preds)

        # Collect IDs
        # The dataset object stores ids in self.ids
        all_ids.extend(test_data["ids"])

    # Concatenate
    final_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 3)

    # Verify Inference Shape
    assert final_preds.shape == (MINI_SIZE, Config.SEQ_LEN, 3)
    print(f"    Inference output shape: {final_preds.shape}")

    # Format Submission
    # We need to flatten predictions: one row per sequence position
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Note: The model predicts [reactivity, deg_Mg_pH10, deg_Mg_50C] (3 cols)
    # The submission requires 5 columns. We fill missing ones with 0 or model logic.
    # For this demo, we assume missing cols are 0.

    submission_rows = []
    target_cols_model = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    for i, sample_id in enumerate(all_ids):
        pred_matrix = final_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Map model outputs to submission columns
            # Model: 0->reactivity, 1->deg_Mg_pH10, 2->deg_Mg_50C
            val_reactivity = pred_matrix[seqpos, 0]
            val_deg_Mg_pH10 = pred_matrix[seqpos, 1]
            val_deg_Mg_50C = pred_matrix[seqpos, 2]

            # Missing columns in model
            val_deg_pH10 = 0.0
            val_deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": val_reactivity,
                    "deg_Mg_pH10": val_deg_Mg_pH10,
                    "deg_pH10": val_deg_pH10,
                    "deg_Mg_50C": val_deg_Mg_50C,
                    "deg_50C": val_deg_50C,
                }
            )

    df_sub = pd.DataFrame(submission_rows)

    # Verify Submission DataFrame
    expected_rows = MINI_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"
    assert "id_seqpos" in df_sub.columns

    # Save to working dir
    sub_path = os.path.join(Config.WORKING_DIR, "submission", "submission.csv")
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    df_sub.to_csv(sub_path, index=False)

    print(f"    Submission file generated at: {sub_path}")
    print(f"    Submission head:\n{df_sub.head(2)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
