import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import library components
from library.config import DEVICE, WORKING_DIR, setup_directories, BATCH_SIZE, IMG_SIZE
from library.utils import seed_everything
from library.data_loader import get_fold_dataloaders, get_test_dataloader
from library.model import MultiScaleRepVGG, reparameterize_model
from library.engine import train_one_epoch, evaluate, make_submission, SWAHandler


def main():
    print("Starting demonstration script...")

    # 1. Setup Environment
    # -------------------------------------------------------------------------
    print("\n[1] Setting up directories and seeds...")
    setup_directories()
    seed_everything(42)

    # Ensure we are using the correct device
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data...")
    # We use fold 0 for demonstration. load_cached_data=True will use existing cache if available.
    train_loader, val_loader = get_fold_dataloaders(fold_idx=0, load_cached_data=True)
    test_loader, test_ids = get_test_dataloader(load_cached_data=True)

    # Validation: Check Data Loaders
    try:
        train_batch, train_labels = next(iter(train_loader))
        print(f"Train Batch Shape: {train_batch.shape}")
        print(f"Train Labels Shape: {train_labels.shape}")

        # Assertions to ensure data pipeline is correct
        assert train_batch.shape == (
            BATCH_SIZE,
            3,
            IMG_SIZE,
            IMG_SIZE,
        ), "Incorrect train batch shape"
        assert train_labels.shape == (BATCH_SIZE,), "Incorrect train label shape"
        assert isinstance(train_batch, torch.Tensor), "Images should be tensors"
        assert isinstance(train_labels, torch.Tensor), "Labels should be tensors"

        print("Data Loading verified.")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[3] Instantiating Model...")
    # Initialize the custom RepVGG model
    model = MultiScaleRepVGG(num_classes=1, deploy=False)
    model.to(DEVICE)

    # Validation: Forward Pass
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        outputs = model(dummy_input)

    # MultiScaleRepVGG returns a list of outputs [out1, out2, out3] for deep supervision
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 3, "Model should return 3 outputs (multi-scale)"
    assert outputs[0].shape == (2, 1), "Output shape mismatch"
    print("Model instantiation and forward pass verified.")

    # 4. Training Loop (Single Epoch)
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train for one epoch
    # Note: train_one_epoch handles Mixup augmentation internally
    avg_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE, epoch=1
    )

    assert isinstance(avg_loss, float), "Loss should be a float"
    assert avg_loss > 0, "Loss should be positive"
    print(f"Training verified. Average Loss: {avg_loss:.4f}")

    # 5. Evaluation
    # -------------------------------------------------------------------------
    print("\n[5] Running Evaluation...")
    # Evaluate on validation set
    val_loss, val_auc, val_preds = evaluate(model, val_loader, criterion, DEVICE)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1"
    assert len(val_preds) == len(val_loader.dataset), "Prediction count mismatch"
    print("Evaluation verified.")

    # 6. SWA (Stochastic Weight Averaging)
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating SWA...")
    swa_handler = SWAHandler(model, optimizer, swa_lr=5e-4)

    # Perform one SWA update (typically done at the end of each epoch in the SWA phase)
    swa_handler.update()

    # Update BatchNorm statistics
    # This runs a pass over the train loader to update running_mean/var for the averaged model
    print("Updating SWA BatchNorm statistics (this may take a few seconds)...")
    swa_handler.update_bn(train_loader, DEVICE)

    swa_model = swa_handler.get_averaged_model()
    assert isinstance(swa_model, torch.nn.Module), "SWA model is not a module"
    print("SWA verified.")

    # 7. Reparameterization (Inference Optimization)
    # -------------------------------------------------------------------------
    print("\n[7] Demonstrating Reparameterization...")
    # We reparameterize the base model for demonstration.
    # This converts multi-branch blocks (Identity + 3x3 + 1x1) to single-branch 3x3 blocks.
    # This modifies the model in-place.
    deploy_model = reparameterize_model(model)
    deploy_model.eval()

    # Verify structure change
    # The 'stem' is a RepVGGBlock. In deploy mode, it should have 'rbr_reparam' and no 'rbr_dense'
    stem_block = deploy_model.stem
    assert hasattr(
        stem_block, "rbr_reparam"
    ), "Block not reparameterized (missing rbr_reparam)"
    assert not hasattr(
        stem_block, "rbr_dense"
    ), "Block not reparameterized (still has rbr_dense)"

    # Verify inference still works
    with torch.no_grad():
        deploy_out = deploy_model(dummy_input)
    assert len(deploy_out) == 3
    print("Reparameterization verified.")

    # 8. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[8] Generating Submission...")
    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    make_submission(deploy_model, test_loader, test_ids, submission_path, DEVICE)

    # Verify file existence and format
    assert os.path.exists(submission_path), "Submission file not created"

    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Submission columns mismatch"
    assert df_sub["has_cactus"].dtype == float, "Prediction column should be float"

    print(f"Submission verified at {submission_path}")
    print("\nDemonstration complete.")


if __name__ == "__main__":
    main()
