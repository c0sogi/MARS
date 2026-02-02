import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Filter warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
# We assume the library files are in the python path or current directory structure
try:
    from library.config import Hyperparameters
    from library.utils import seed_everything, MCRMSELoss
    from library.data import get_dataloaders
    from library.model import RNAModel
    from library.engine import fit
except ImportError as e:
    print(f"Error importing library modules: {e}")
    sys.exit(1)


def main():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Hyperparameters for a fast demonstration run
    print("Configuring hyperparameters for demo...")
    Hyperparameters.NUM_EPOCHS = 2
    Hyperparameters.DATA_SUBSET_FRACTION = 0.1  # Use 10% of data for speed
    Hyperparameters.BATCH_SIZE = 16

    # Set a specific working directory for this demo to avoid overwriting real work
    Hyperparameters.WORKING_DIR = "./working/demo_execution/"
    Hyperparameters.CACHE_DIR = os.path.join(Hyperparameters.WORKING_DIR, "cache")
    Hyperparameters.MODELS_DIR = os.path.join(Hyperparameters.WORKING_DIR, "models")

    # Ensure directories exist (Config usually does this, but we changed paths)
    os.makedirs(Hyperparameters.CACHE_DIR, exist_ok=True)
    os.makedirs(Hyperparameters.MODELS_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Hyperparameters.SEED)
    device = Hyperparameters.DEVICE
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nInitializing DataLoaders...")
    # This will load metadata, process features, and cache them in the new demo dir
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data_flag=True)

    # Verify Data Shapes
    print("Verifying data shapes...")
    sample_batch = next(iter(train_loader))
    inputs = sample_batch["inputs"]
    adjacency = sample_batch["adjacency"]
    targets = sample_batch["targets"]

    # Expected: (Batch, 107, 14)
    assert inputs.shape[1] == 107, f"Expected seq len 107, got {inputs.shape[1]}"
    assert inputs.shape[2] == 14, f"Expected input dim 14, got {inputs.shape[2]}"
    # Expected: (Batch, 107)
    assert adjacency.shape[1] == 107, "Adjacency map length mismatch"
    # Expected: (Batch, 68, 5) - Targets are only for first 68 positions
    assert targets.shape[1] == 68, f"Expected target seq len 68, got {targets.shape[1]}"
    assert targets.shape[2] == 5, f"Expected 5 targets, got {targets.shape[2]}"

    print("Data shapes verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = RNAModel().to(device)

    # Verify Model Forward Pass
    with torch.no_grad():
        dummy_out = model(inputs.to(device), adjacency.to(device))

    # Output should be (Batch, 107, 5) - Model predicts for full sequence
    assert dummy_out.shape == (
        inputs.shape[0],
        107,
        5,
    ), f"Model output shape mismatch. Expected {(inputs.shape[0], 107, 5)}, got {dummy_out.shape}"
    print("Model forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("\nStarting Training Loop...")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Hyperparameters.LEARNING_RATE,
        weight_decay=Hyperparameters.WEIGHT_DECAY,
    )

    criterion = MCRMSELoss()

    model_save_path = os.path.join(Hyperparameters.WORKING_DIR, "best_model.pth")

    best_score = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=Hyperparameters.NUM_EPOCHS,
        patience=2,  # Short patience for demo
        model_save_path=model_save_path,
    )

    print(f"Training complete. Best Validation Score: {best_score:.4f}")
    assert os.path.exists(model_save_path), "Model file was not saved."

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\nGenerating Predictions on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    with torch.no_grad():
        for batch in test_loader:
            t_inputs = batch["inputs"].to(device)
            t_adj = batch["adjacency"].to(device)
            t_ids = batch["id"]  # List of strings

            # Forward pass: (B, 107, 5)
            outputs = model(t_inputs, t_adj)
            outputs = outputs.cpu().numpy()

            # Process batch
            batch_size = outputs.shape[0]
            seq_len = outputs.shape[1]

            for i in range(batch_size):
                sample_id = t_ids[i]
                sample_preds = outputs[i]  # (107, 5)

                # Create rows for submission
                for seq_pos in range(seq_len):
                    row_id = f"{sample_id}_{seq_pos}"
                    preds = sample_preds[seq_pos]

                    row_dict = {"id_seqpos": row_id}
                    for col_idx, col_name in enumerate(target_cols):
                        row_dict[col_name] = float(preds[col_idx])

                    submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Verify Submission Format
    print("Verifying submission format...")
    expected_cols = ["id_seqpos"] + target_cols
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    # Check if we have rows
    assert len(submission_df) > 0, "Submission DataFrame is empty"

    # Save Submission
    submission_path = os.path.join(Hyperparameters.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
