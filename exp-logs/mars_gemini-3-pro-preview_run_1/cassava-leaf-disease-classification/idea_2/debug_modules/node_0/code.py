import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging

# Import provided library modules
from library import config
from library.config import CFG
from library import utils
from library import dataset
from library import model
from library import loss
from library import engine
from library import inference


def run_demonstration():
    print("=== Starting Cassava Leaf Disease Classification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1/5] Setting up configuration...")

    # Set seeds for reproducibility
    utils.seed_everything(42)

    # Define temporary working directory for this demo
    DEMO_WORK_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR)

    # Override CFG parameters for speed and demonstration purposes
    CFG.debug = True
    CFG.output_dir = DEMO_WORK_DIR
    CFG.model_save_name = "demo_best_model.pth"
    CFG.epochs = 1
    CFG.batch_size = 4
    CFG.num_workers = 0  # Disable multiprocessing for simple script execution
    CFG.print_freq = 5
    # Use a smaller image size if desired, but we keep 380 to match model config
    # We disable pretrained weights to avoid downloading large files during demo
    CFG.pretrained = False

    print(f"Working directory set to: {CFG.output_dir}")

    # -------------------------------------------------------------------------
    # 2. Dataset & Transforms Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Dataset and Transforms...")

    # Load training metadata
    df_train_full = pd.read_csv(CFG.train_csv)

    # Select a tiny subset (16 samples) to simulate a dataset
    df_subset = df_train_full.iloc[:16].reset_index(drop=True)
    print(f"Created subset dataframe with {len(df_subset)} samples.")

    # Instantiate Dataset
    train_ds = dataset.CassavaDataset(
        df_subset, transform=dataset.get_transforms("train"), output_label=True
    )

    # Verify single item retrieval
    img, label = train_ds[0]
    print(f"Single item - Image Shape: {img.shape}, Label: {label}")

    assert img.shape == (
        3,
        CFG.image_size,
        CFG.image_size,
    ), f"Expected image shape (3, {CFG.image_size}, {CFG.image_size}), got {img.shape}"
    assert isinstance(label, (int, np.integer)), "Label must be an integer"

    # Verify CollateMixupCutmix
    # We force mix_p=1.0 to ensure mixing logic is triggered and tested
    collate_fn = dataset.CollateMixupCutmix(
        mix_p=1.0, alpha=0.4, n_classes=CFG.target_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
    )

    # Fetch a batch to verify shapes and soft targets
    batch_imgs, batch_targets = next(iter(train_loader))
    print(
        f"Batch - Images Shape: {batch_imgs.shape}, Targets Shape: {batch_targets.shape}"
    )

    assert batch_imgs.shape == (
        CFG.batch_size,
        3,
        CFG.image_size,
        CFG.image_size,
    ), "Batch image shape mismatch"
    assert batch_targets.shape == (
        CFG.batch_size,
        CFG.target_size,
    ), "Batch target shape mismatch"

    # Verify targets sum to 1 (Soft targets)
    sums = batch_targets.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums)), "Soft targets should sum to 1"

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model and Loss...")

    device = CFG.device

    # Instantiate Model (pretrained=False for speed)
    net = model.CassavaModel(model_name=CFG.model_name, pretrained=False)
    net.to(device)

    # Forward pass verification
    batch_imgs = batch_imgs.to(device)
    batch_targets = batch_targets.to(device)

    with torch.no_grad():
        logits = net(batch_imgs)

    print(f"Model Output (Logits) Shape: {logits.shape}")
    assert logits.shape == (CFG.batch_size, CFG.target_size), "Logits shape mismatch"

    # Loss verification
    criterion = loss.SoftTargetCrossEntropy()
    loss_val = criterion(logits, batch_targets)

    print(f"Calculated Loss: {loss_val.item():.4f}")
    assert loss_val.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss_val), "Loss is NaN"

    # -------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[4/5] Simulating Training Loop (1 Epoch)...")

    # Setup Logger
    logger = utils.init_logger(os.path.join(CFG.output_dir, "demo.log"))

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=CFG.epochs, steps_per_epoch=len(train_loader)
    )

    # Setup Validation Loader (using same subset for convenience)
    val_ds = dataset.CassavaDataset(
        df_subset, transform=dataset.get_transforms("valid"), output_label=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers
    )

    # Run Training Engine
    engine.fit(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=CFG.epochs,
        logger=logger,
        patience=1,
    )

    # Verify Model Checkpoint
    checkpoint_path = os.path.join(CFG.output_dir, CFG.model_save_name)
    if os.path.exists(checkpoint_path):
        print(f"SUCCESS: Model checkpoint saved at {checkpoint_path}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved after training.")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Inference Pipeline...")

    # Create a dummy test subset metadata file
    df_test_full = pd.read_csv(CFG.test_csv)
    df_test_subset = df_test_full.iloc[:8].copy()  # Take 8 test images
    test_subset_path = os.path.join(CFG.output_dir, "test_subset.csv")
    df_test_subset.to_csv(test_subset_path, index=False)

    # Point CFG to this new test file
    CFG.test_csv = test_subset_path

    # Run Inference
    # This function loads the model from CFG.output_dir/CFG.model_save_name
    # and saves predictions to ./submission/submission.csv
    try:
        inference.run_inference()
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    # Verify Submission
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission file generated with shape: {sub_df.shape}")
        print(sub_df.head())

        assert len(sub_df) == 8, "Submission row count mismatch"
        assert list(sub_df.columns) == [
            "image_id",
            "label",
        ], "Submission columns mismatch"
        assert sub_df["label"].dtype == "int64", "Label column should be integer"
        print("SUCCESS: Submission file is valid.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== All Validations Passed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
