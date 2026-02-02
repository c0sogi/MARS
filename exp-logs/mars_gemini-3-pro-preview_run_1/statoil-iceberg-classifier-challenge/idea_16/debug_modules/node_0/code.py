import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import IcebergResNet
from library.engine import IcebergTrainer
from library.pseudo_labeling import generate_pseudo_labels


def main():
    print("=== Iceberg Classification Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Override Config for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = Config.WORKING_DIR

    # Speed optimizations
    Config.MAX_EPOCHS = 2  # Train for only 2 epochs
    Config.SWA_EPOCHS = 1  # 1 Epoch of SWA
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Relax SSL thresholds to ensure pseudo-labels are generated with a weak demo model
    Config.SSL_CONFIDENCE_HIGH = 0.51
    Config.SSL_CONFIDENCE_LOW = 0.49
    Config.SSL_STD_THRESHOLD = 1.0

    # Initialize directories
    Config.setup()

    # Set random seed for reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data...")

    # Load dataloaders (force processing first time by disabling cache loading initially)
    # In a real run, load_cached_data=True is preferred.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verification: Check batch structure
    images, angles, labels = next(iter(train_loader))
    print(
        f"    Train Batch - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect image tensor shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")

    device = Config.DEVICE
    # Use pretrained=False to avoid downloading weights during demo
    model = IcebergResNet(pretrained=False)
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_out = model(images.to(device), angles.to(device))
    assert dummy_out.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("    Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Phase 1: Base Training
    # -------------------------------------------------------------------------
    print("\n[4] Training Base Model...")

    trainer = IcebergTrainer(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Train with SWA
    # We set swa_start_epoch=2 so SWA happens at the very end of our 2-epoch run
    trained_model = trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.MAX_EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_name="best_model.pth",
        use_swa=True,
        swa_start_epoch=2,
    )

    # Verification: Checkpoints
    best_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    swa_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model_swa.pth")

    if os.path.exists(best_ckpt):
        print(f"    Saved best model: {best_ckpt}")
    if os.path.exists(swa_ckpt):
        print(f"    Saved SWA model: {swa_ckpt}")

    assert os.path.exists(swa_ckpt), "SWA checkpoint was not created."

    # -------------------------------------------------------------------------
    # 5. Pseudo-Labeling (SSL)
    # -------------------------------------------------------------------------
    print("\n[5] Generating Pseudo-Labels...")

    # Simulate an ensemble by using the same SWA model twice
    # In practice, you would use different folds or seeds.
    ensemble_paths = [swa_ckpt, swa_ckpt]

    # Generate labels on a small subset (debug_limit) for speed
    pseudo_imgs, pseudo_angs, pseudo_lbls = generate_pseudo_labels(
        model_paths=ensemble_paths,
        load_cached_data=False,
        debug_limit=50,  # Only process 50 test images
    )

    print(f"    Generated {len(pseudo_imgs)} pseudo-labels.")

    # -------------------------------------------------------------------------
    # 6. Phase 2: Retraining with SSL
    # -------------------------------------------------------------------------
    print("\n[6] Retraining with Pseudo-Labels...")

    if len(pseudo_imgs) > 0:
        extra_data = (pseudo_imgs, pseudo_angs, pseudo_lbls)

        # Get new train loader augmented with pseudo-labels
        ssl_train_loader, _, _ = get_dataloaders(
            load_cached_data=True, extra_data=extra_data
        )

        # Verification: Dataset size
        base_len = len(train_loader.dataset)
        ssl_len = len(ssl_train_loader.dataset)
        print(f"    Base Dataset: {base_len}, Augmented Dataset: {ssl_len}")
        assert ssl_len == base_len + len(pseudo_imgs), "Dataset augmentation failed"

        # Fine-tune for 1 epoch
        trainer.fit(
            ssl_train_loader,
            val_loader,
            epochs=1,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_name="ssl_finetuned.pth",
            use_swa=False,
        )
    else:
        print("    Skipping SSL training (no pseudo-labels generated).")

    # -------------------------------------------------------------------------
    # 7. Prediction & Submission
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission...")

    # Predict on full test set
    preds = trainer.predict(test_loader)

    # Verification: Prediction range
    assert len(preds) == len(test_loader.dataset), "Prediction count mismatch"
    assert (preds >= 0.0).all() and (
        preds <= 1.0
    ).all(), "Predictions out of probability range [0, 1]"

    # Create Submission DataFrame
    test_ids = test_loader.dataset.ids
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": preds})

    # Save
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"    Submission saved to: {sub_path}")
    print(f"    Shape: {submission_df.shape}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
