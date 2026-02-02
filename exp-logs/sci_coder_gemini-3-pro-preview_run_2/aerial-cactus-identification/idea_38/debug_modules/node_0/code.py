import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library components
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import UltraWideRepRes2NeXt
from library.engine import train_one_epoch, evaluate
from library.inference import predict_ensemble


def main():
    print("=== Cactus Identification: Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    # Ensure reproducibility
    set_seed(42)
    device = get_device()

    # Define working paths
    working_dir = "./working/demo_execution"
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_path = os.path.join(working_dir, "model_checkpoint.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ---------------------------------------------------------
    # 2. Data Loading (library.dataset)
    # ---------------------------------------------------------
    print("\n[Data] Initializing DataLoaders...")

    # We use a small batch size and debug_size to ensure the demo runs quickly
    BATCH_SIZE = 8
    DEBUG_SIZE = 32

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=True,  # Use cache if available for speed
        debug_size=DEBUG_SIZE,  # Limit dataset size
    )

    # Verify Data Loading
    images, targets = next(iter(train_loader))
    print(f"       Batch Image Shape: {images.shape}")
    print(f"       Batch Target Shape: {targets.shape}")

    # Assertions
    assert images.shape == (BATCH_SIZE, 3, 32, 32), "Unexpected image batch dimensions."
    assert targets.shape == (BATCH_SIZE,), "Unexpected target batch dimensions."
    assert (
        len(test_loader.dataset) == DEBUG_SIZE
    ), "Test dataset size does not match debug_size."

    # ---------------------------------------------------------
    # 3. Model Initialization (library.model)
    # ---------------------------------------------------------
    print("\n[Model] Instantiating UltraWideRepRes2NeXt...")

    # Initialize model in training mode (deploy=False keeps multi-branch blocks)
    model = UltraWideRepRes2NeXt(num_classes=1, deploy=False).to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"       Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1), "Model output shape mismatch."

    # ---------------------------------------------------------
    # 4. Training Loop (library.engine)
    # ---------------------------------------------------------
    print("\n[Train] Running training simulation (1 Epoch)...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    train_loss, train_auc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"       Train | Loss: {train_loss:.4f} | AUC: {train_auc:.4f}")

    # Evaluate
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    print(f"       Val   | Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")

    # Verify metrics are valid numbers
    assert not np.isnan(train_loss), "Training loss returned NaN."
    assert not np.isnan(val_loss), "Validation loss returned NaN."

    # ---------------------------------------------------------
    # 5. Checkpointing
    # ---------------------------------------------------------
    print(f"\n[Save] Saving model state to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    # ---------------------------------------------------------
    # 6. Inference and TTA (library.inference)
    # ---------------------------------------------------------
    print("\n[Infer] Running Ensemble Prediction with TTA...")

    # predict_ensemble handles:
    # 1. Loading the model from checkpoint
    # 2. Structural Re-parameterization (switch_to_deploy)
    # 3. Test Time Augmentation (Horizontal/Vertical Flips)
    # 4. Aggregation and CSV generation

    df_result = predict_ensemble(
        model_paths=[checkpoint_path],
        test_loader=test_loader,
        device=device,
        output_path=submission_path,
    )

    # ---------------------------------------------------------
    # 7. Validation of Results
    # ---------------------------------------------------------
    print("\n[Verify] Validating submission file...")

    # Check file existence
    assert os.path.exists(submission_path), "Submission CSV not found."

    # Load and check content
    df_loaded = pd.read_csv(submission_path)
    print(f"       Submission Rows: {len(df_loaded)}")
    print(f"       Columns: {list(df_loaded.columns)}")

    # Assertions
    assert (
        len(df_loaded) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} rows, found {len(df_loaded)}."
    assert "id" in df_loaded.columns, "Column 'id' missing."
    assert "has_cactus" in df_loaded.columns, "Column 'has_cactus' missing."

    # Check probability range
    probs = df_loaded["has_cactus"].values
    assert np.all(
        (probs >= 0.0) & (probs <= 1.0)
    ), "Predictions contain values outside [0, 1]."

    print("\n=== Demonstration Complete: All checks passed. ===")


if __name__ == "__main__":
    main()
