import os
import torch
import pandas as pd
import numpy as np

# Import library components
import library.config as config
import library.dataset as dataset_module
from library.dataset import get_dataloaders
from library.model import HybridRNNTransformer
from library.loss import SignalWeightedMSELoss
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    print("=== RNA Degradation Prediction Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    # We monkey-patch the DEBUG flags in the imported modules to force
    # the use of a small data subset (50 samples) for this demo.
    config.DEBUG = True
    dataset_module.DEBUG = True

    config.DEBUG_SUBSET_SIZE = 50
    dataset_module.DEBUG_SUBSET_SIZE = 50

    # Define demo hyperparameters
    DEMO_BATCH_SIZE = 8
    DEMO_EPOCHS = 2

    # Set random seeds for reproducibility
    config.seed_everything(42)
    print(
        f"Configuration: DEBUG=True, Subset Size={config.DEBUG_SUBSET_SIZE}, Epochs={DEMO_EPOCHS}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and Processing Data...")

    # load_cached_data=False forces the dataset to be re-processed from source.
    # Because DEBUG is True, it will slice the source DataFrames to 50 rows
    # before processing and saving to the cache in ./working/idea_5/
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, batch_size=DEMO_BATCH_SIZE
    )

    # Verify Data Integrity
    print("   Verifying data batch structure...")
    batch = next(iter(train_loader))

    # Check for required keys
    required_keys = [
        "id",
        "sequence",
        "structure",
        "predicted_loop_type",
        "targets",
        "mask",
        "weight",
    ]
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Batch missing key: {key}")

    # Check tensor shapes
    # Sequence: (Batch, 107)
    assert batch["sequence"].shape == (
        DEMO_BATCH_SIZE,
        config.SEQ_LENGTH,
    ), f"Sequence shape mismatch: {batch['sequence'].shape}"
    # Targets: (Batch, 107, 5)
    assert batch["targets"].shape == (
        DEMO_BATCH_SIZE,
        config.SEQ_LENGTH,
        5,
    ), f"Targets shape mismatch: {batch['targets'].shape}"

    print("   Data loaded and shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = HybridRNNTransformer().to(device)

    # Verify Forward Pass
    print("   Verifying forward pass...")
    seq = batch["sequence"].to(device)
    struct = batch["structure"].to(device)
    loop = batch["predicted_loop_type"].to(device)

    with torch.no_grad():
        outputs = model(seq, struct, loop)

    # Output should be (Batch, Seq_Len, Num_Targets)
    assert outputs.shape == (
        DEMO_BATCH_SIZE,
        config.SEQ_LENGTH,
        5,
    ), f"Model output shape mismatch: {outputs.shape}"
    print("   Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Loss Function...")
    criterion = SignalWeightedMSELoss()

    targets = batch["targets"].to(device)
    masks = batch["mask"].to(device)
    weights = batch["weight"].to(device)

    loss = criterion(outputs, targets, masks, weights)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"   Loss calculated successfully: {loss.item():.6f}")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(model, device, criterion, optimizer)

    # Run training for a limited number of epochs
    trainer.fit(train_loader, val_loader, epochs=DEMO_EPOCHS)

    # Check if the best model was saved
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Expected model checkpoint at {best_model_path}")
    print("   Training complete. Best model checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Submission...")
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Generate predictions using the trained model
    # This uses the test_loader which was also restricted to 50 samples by the DEBUG flag
    generate_submission(
        model_path=best_model_path,
        output_path=submission_path,
        batch_size=DEMO_BATCH_SIZE,
    )

    # Verify Submission File
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    print(f"   Submission loaded. Shape: {df_sub.shape}")

    # Expected rows: Subset_Size * Seq_Length
    expected_rows = config.DEBUG_SUBSET_SIZE * config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print("   Submission content verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
