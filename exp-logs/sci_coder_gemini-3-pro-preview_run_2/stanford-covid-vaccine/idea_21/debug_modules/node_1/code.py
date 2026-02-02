import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, DEVICE
from library.utils import set_seed, GlobalMCRMSE
from library.data import get_dataloaders
from library.model import ScaleDecoupledDenseNet
from library.loss import MaskedMCRMSELoss
from library.train import run_training


def main():
    print("=== Starting Library Demonstration Script ===\n")

    # 1. Setup and Configuration Override
    # We override the CACHE_DIR to keep this demo execution isolated.
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch the Config class to use the demo directory
    Config.CACHE_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Set seed for reproducibility
    set_seed(42)
    print(f"Configuration: Device={DEVICE}, Cache Dir={Config.CACHE_DIR}")

    # 2. Data Loading Verification
    print("\n[Step 1] Verifying Data Loading...")
    batch_size = 4
    max_samples = 12  # Small subset for speed

    # We force reprocessing (load_cached_data=False) to ensure data logic works
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=False, max_samples=max_samples
    )

    # Fetch one batch
    inputs, partner_indices, targets = next(iter(train_loader))

    # Verify shapes
    # Inputs: (Batch, Channels=19, SeqLen=107)
    assert inputs.shape == (
        batch_size,
        Config.INPUT_CHANNELS,
        Config.SEQ_LEN,
    ), f"Input shape mismatch. Expected {(batch_size, 19, 107)}, got {inputs.shape}"

    # Partner Indices: (Batch, SeqLen=107)
    assert partner_indices.shape == (
        batch_size,
        Config.SEQ_LEN,
    ), f"Partner indices shape mismatch. Expected {(batch_size, 107)}, got {partner_indices.shape}"

    # Targets: (Batch, SeqLen=107, 5)
    assert targets.shape == (
        batch_size,
        Config.SEQ_LEN,
        5,
    ), f"Targets shape mismatch. Expected {(batch_size, 107, 5)}, got {targets.shape}"

    print("Data loading verification passed. Shapes are correct.")

    # 3. Model Forward Pass Verification
    print("\n[Step 2] Verifying Model Architecture...")
    model = ScaleDecoupledDenseNet().to(DEVICE)

    # Move batch to device
    inputs = inputs.to(DEVICE)
    partner_indices = partner_indices.to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(inputs, partner_indices)

    # Verify output shape: (Batch, SeqLen, 5)
    assert preds.shape == (
        batch_size,
        Config.SEQ_LEN,
        5,
    ), f"Model output shape mismatch. Expected {(batch_size, 107, 5)}, got {preds.shape}"

    print("Model forward pass verification passed.")

    # 4. Loss Function Logic Verification
    print("\n[Step 3] Verifying Masked MCRMSE Loss Logic...")
    criterion = MaskedMCRMSELoss()

    # Create dummy data
    # Scored columns are 0, 1, 3. Columns 2 and 4 are ignored.
    # We will create a scenario where only column 0 has error.

    # B=2, L=10, C=5
    dummy_targets = torch.zeros((2, 10, 5))
    dummy_preds = torch.zeros((2, 10, 5))

    # Set error of 1.0 on column 0 for all positions
    dummy_preds[:, :, 0] = 1.0
    # Set error of 100.0 on column 2 (ignored), should not affect loss
    dummy_preds[:, :, 2] = 100.0

    # Calculation:
    # Col 0: MSE = mean((1-0)^2) = 1.0 -> RMSE = 1.0
    # Col 1: MSE = 0.0 -> RMSE = 0.0
    # Col 3: MSE = 0.0 -> RMSE = 0.0
    # MCRMSE = mean([1.0, 0.0, 0.0]) = 0.3333...

    loss_val = criterion(dummy_preds, dummy_targets)
    expected_val = 1.0 / 3.0

    assert np.isclose(
        loss_val.item(), expected_val, atol=1e-5
    ), f"Loss calculation incorrect. Expected {expected_val:.5f}, got {loss_val.item():.5f}"

    print("Loss function verification passed.")

    # 5. Full Training Loop Execution
    print("\n[Step 4] Executing Training Loop (Demo)...")

    # Run training for 2 epochs on 50 samples
    # This uses the library.train.run_training function
    best_score = run_training(max_epochs=2, max_samples=50)

    print(f"Training completed. Best Validation MCRMSE: {best_score:.5f}")

    # Verify model checkpoint exists
    checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not saved."
    print(f"Checkpoint verified at {checkpoint_path}")

    # 6. Inference and Submission Generation
    print("\n[Step 5] Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Get test loader (already initialized in step 2, but let's ensure we use the test set)
    # We reuse the test_loader created earlier which has max_samples=12

    all_preds = []
    all_ids = []

    # Need to retrieve IDs. The loader returns (inputs, partners, targets),
    # but the dataset has the IDs. We can access the dataset from the loader.
    dataset_ids = test_loader.dataset.ids

    with torch.no_grad():
        for i, (inputs, partner_indices, _) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            partner_indices = partner_indices.to(DEVICE)

            batch_preds = model(inputs, partner_indices)
            all_preds.append(batch_preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Generate CSV rows
    submission_rows = []
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(dataset_ids):
        # Ensure we don't go out of bounds if loader dropped last or mismatched
        if i >= len(all_preds):
            break

        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(cols):
                row_dict[col_name] = row_vals[col_idx]
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission generated with {len(sub_df)} rows.")

    # Verify Submission Format
    assert sub_df.shape[1] == 6, "Submission should have 6 columns"
    assert "id_seqpos" in sub_df.columns, "Missing id_seqpos column"
    assert sub_df["id_seqpos"].iloc[0].startswith("id_"), "Invalid ID format"

    print("Submission format verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
