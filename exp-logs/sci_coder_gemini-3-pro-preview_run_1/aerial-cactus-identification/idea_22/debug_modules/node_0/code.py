import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, Mixup
from library.dataset import load_dataset_to_memory, get_fold_loaders, get_test_loader
from library.model import CactusRepVGG, RepVGGBlock
from library.engine import train_one_epoch, validate, SWAHandler, generate_submission

if __name__ == "__main__":
    print("Starting Cactus Identification Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use only 100 images
    Config.BATCH_SIZE = 16  # Smaller batch size for small subset
    Config.EPOCHS_CONVERGENCE = 1  # Only 1 epoch for standard training
    Config.EPOCHS_SWA = 1  # Only 1 epoch for SWA
    Config.TOTAL_EPOCHS = Config.EPOCHS_CONVERGENCE + Config.EPOCHS_SWA
    Config.SWA_START_EPOCH = Config.EPOCHS_CONVERGENCE
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory for this demo exists
    demo_working_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    os.makedirs(demo_working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    Config.print_config()

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("\n[Step 2] Loading Data...")
    # Force reload to demonstrate loading logic (ignoring existing cache for this run)
    train_imgs, train_labels, test_imgs, test_ids = load_dataset_to_memory(
        load_cached_data=False
    )

    # Validate Data Shapes
    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")

    # Assertions to verify Debug mode worked
    assert (
        train_imgs.shape[0] == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} train images in DEBUG mode, got {train_imgs.shape[0]}"
    assert len(train_labels) == Config.DEBUG_SUBSET_SIZE
    assert test_imgs.shape[0] == Config.DEBUG_SUBSET_SIZE

    # Get Loaders for Fold 0
    print("Creating DataLoaders for Fold 0...")
    train_loader, val_loader = get_fold_loaders(0, (train_imgs, train_labels))

    # Verify Loader
    sample_batch, sample_labels = next(iter(train_loader))
    assert sample_batch.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect batch shape"
    assert sample_labels.shape == (Config.BATCH_SIZE,), "Incorrect label shape"
    print("DataLoaders created and verified.")

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("\n[Step 3] Initializing Model...")
    device = Config.DEVICE
    model = CactusRepVGG(num_classes=1).to(device)

    # Verify Forward Pass (Dual Head Output)
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    out_tex, out_sem = model(dummy_input)

    assert out_tex.shape == (2, 1), "Texture head output shape mismatch"
    assert out_sem.shape == (2, 1), "Semantic head output shape mismatch"
    print("Model initialized and forward pass verified.")

    # =========================================================================
    # 4. Training Loop Simulation
    # =========================================================================
    print("\n[Step 4] Running Training Simulation...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA, device=device)
    swa_handler = SWAHandler(model, device, swa_start_epoch=Config.SWA_START_EPOCH)

    for epoch in range(Config.TOTAL_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mixup_fn
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # SWA Update
        swa_handler.update(model, epoch)

    # Finalize SWA (Update BN statistics)
    print("Updating SWA Batch Normalization statistics...")
    swa_handler.update_bn(train_loader)
    final_model = swa_handler.get_model()
    print("Training simulation complete.")

    # =========================================================================
    # 5. Inference & Submission
    # =========================================================================
    print("\n[Step 5] Generating Submission...")

    test_loader = get_test_loader(
        test_imgs, batch_size=Config.BATCH_SIZE, num_workers=0
    )
    submission_path = os.path.join(demo_working_dir, "demo_submission.csv")

    # Generate submission using the SWA model
    # Note: generate_submission expects a list of models (ensemble), so we pass [final_model]
    generate_submission([final_model], test_loader, test_ids, device, submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns mismatch"
    assert len(df_sub) == Config.DEBUG_SUBSET_SIZE, "Submission row count mismatch"
    print(f"Submission verified at {submission_path}")

    # =========================================================================
    # 6. RepVGG Deployment Mode Verification
    # =========================================================================
    print("\n[Step 6] Verifying RepVGG Deployment Mode...")

    # Access the underlying model inside AveragedModel if using SWA, or the base model
    # Here we use the base model 'model' which is still in training mode

    # Check structure before deploy (Should have rbr_dense, rbr_1x1)
    # We inspect the stem block
    assert hasattr(
        model.stem, "rbr_dense"
    ), "Model should have dense branch before deploy"
    assert not hasattr(
        model.stem, "rbr_reparam"
    ), "Model should NOT have reparam branch before deploy"

    print("Switching model to deploy mode...")
    model.switch_to_deploy()

    # Check structure after deploy
    assert not hasattr(
        model.stem, "rbr_dense"
    ), "Model should NOT have dense branch after deploy"
    assert hasattr(
        model.stem, "rbr_reparam"
    ), "Model should have reparam branch after deploy"

    # Verify inference still works
    model.eval()
    with torch.no_grad():
        deploy_out_tex, deploy_out_sem = model(dummy_input)
        assert deploy_out_tex.shape == (2, 1)

    print("RepVGG structural re-parameterization successful.")

    print("\nAll demonstration steps completed successfully.")
