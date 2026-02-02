import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.models as models
import library.engine as engine
import library.inference as inference


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Define a lightweight demo configuration
    # We use resnet18 and a small image size for speed
    DEMO_MODEL_NAME = "resnet18"
    DEMO_IMG_SIZE = 224
    DEMO_BATCH_SIZE = 8
    DEMO_EPOCHS = 1
    DEMO_FOLDS = 2

    demo_config = config.ModelConfig(
        model_name=DEMO_MODEL_NAME,
        img_size=DEMO_IMG_SIZE,
        epochs=DEMO_EPOCHS,
        batch_size=DEMO_BATCH_SIZE,
        num_folds=DEMO_FOLDS,
    )

    # 2. Data Loading & Folds
    print("\n--- Testing Data Loading ---")
    # Load full metadata and create folds
    # We force reload to ensure fresh start
    if os.path.exists(os.path.join(config.WORKING_DIR, "folds.parquet")):
        os.remove(os.path.join(config.WORKING_DIR, "folds.parquet"))

    full_df = dataset.load_data_and_create_folds(
        n_folds=DEMO_FOLDS, load_cached_data=False
    )

    # OPTIMIZATION: Subsample data for speed
    # Take 50 samples for train/val combined
    subset_df = full_df.sample(n=50, random_state=config.SEED).reset_index(drop=True)
    # Re-assign folds to ensure we have data in both fold 0 and fold 1
    subset_df["fold"] = np.random.randint(0, DEMO_FOLDS, size=len(subset_df))
    # Ensure at least one sample in fold 0 for val and train
    subset_df.loc[0, "fold"] = 0  # Val for fold 0
    subset_df.loc[1, "fold"] = 1  # Train for fold 0

    print(f"Subset dataframe shape: {subset_df.shape}")

    # 3. Dataset & DataLoader
    print("\n--- Testing Dataset & DataLoader ---")
    train_loader, val_loader = dataset.get_dataloaders(
        subset_df,
        fold=0,
        image_size=DEMO_IMG_SIZE,
        batch_size=DEMO_BATCH_SIZE,
        num_workers=2,
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Should be [Batch, 3, H, W]
    print(f"Batch Label Shape: {labels.shape}")  # Should be [Batch]

    assert images.shape == (
        DEMO_BATCH_SIZE,
        3,
        DEMO_IMG_SIZE,
        DEMO_IMG_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (DEMO_BATCH_SIZE,), "Incorrect label batch shape"

    # 4. Model Instantiation
    print("\n--- Testing Model Creation ---")
    model = models.get_model(DEMO_MODEL_NAME, pretrained=True, num_classes=1)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        logits = model(images.to(device))
    assert logits.shape == (DEMO_BATCH_SIZE, 1), "Output logits shape mismatch"
    print("Model forward pass successful.")

    # 5. Training Engine (Fit)
    print("\n--- Testing Training Engine (Fit) ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DEMO_EPOCHS)

    # Train for 1 epoch on the subset
    # This will also save the checkpoint to WORKING_DIR
    trained_model = engine.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=DEMO_EPOCHS,
        model_name=demo_config.name,
        fold=0,
    )

    # Verify Checkpoint Existence
    ckpt_path = utils.get_checkpoint_path(demo_config.name, 0)
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print(f"Training complete. Checkpoint saved to {ckpt_path}")

    # 6. Inference
    print("\n--- Testing Inference & Ensemble ---")

    # Monkey Patching: Override the ENSEMBLE_CONFIGS in library.inference
    # to use our demo config. This ensures generate_ensemble_predictions
    # uses the model we just trained and the checkpoint we just saved.
    original_configs = inference.ENSEMBLE_CONFIGS
    inference.ENSEMBLE_CONFIGS = [demo_config]

    # We also need to ensure the test loader uses a small subset or runs quickly.
    # The library uses get_test_loader which reads metadata/test.csv.
    # We cannot modify the CSV file on disk (read-only input), but we can't easily
    # mock the internal call inside inference.py without more complex patching.
    # However, since we are only running 1 model for 1 fold, and the test set is 2500 images,
    # inference should be reasonably fast (approx 10-20 seconds on GPU).

    try:
        # Run inference
        inference.generate_ensemble_predictions(use_tta=False)  # Disable TTA for speed

        # Verify Submission
        submission_path = config.SUBMISSION_PATH
        assert os.path.exists(submission_path), "Submission file was not created"

        sub_df = pd.read_csv(submission_path)
        print(f"Submission file loaded. Shape: {sub_df.shape}")
        print(sub_df.head())

        # Check constraints
        assert (
            "id" in sub_df.columns and "label" in sub_df.columns
        ), "Missing columns in submission"
        assert (
            sub_df["label"].min() >= 0 and sub_df["label"].max() <= 1
        ), "Probabilities out of range"
        assert not sub_df.isnull().values.any(), "Submission contains NaNs"

        print("Inference and submission generation successful.")

    finally:
        # Restore original configs (good practice, though script ends here)
        inference.ENSEMBLE_CONFIGS = original_configs

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
