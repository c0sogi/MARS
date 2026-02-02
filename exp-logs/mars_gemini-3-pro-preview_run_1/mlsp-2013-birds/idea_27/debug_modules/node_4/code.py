import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import from the provided library
from library.config import Config, set_seed
from library.dataset import BirdDataset
from library.model import get_resnet34
from library.training import Trainer
from library.inference import generate_calibrated_pseudo_labels, predict_student
from library.utils import get_logger


def run_demo():
    # 1. Setup and Configuration
    print("--- Step 1: Setup ---")
    set_seed(42)

    # Define demo-specific parameters to override defaults for speed
    DEMO_BATCH_SIZE = 4
    DEMO_EPOCHS = 2
    DEMO_SWA_START = 1
    SUBSET_SIZE = 16  # Small subset for speed

    # Ensure working directory exists (Config creates it, but good to be sure)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    logger = get_logger(os.path.join(Config.WORKING_DIR, "demo.log"))
    logger.info("Starting Library Demo")

    # 2. Dataset and DataLoader
    print("\n--- Step 2: Data Loading ---")
    # Initialize datasets
    full_train_dataset = BirdDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, mode="train"
    )
    full_val_dataset = BirdDataset(metadata_path=Config.VAL_METADATA_PATH, mode="val")

    # Create subsets for rapid execution
    train_indices = list(range(min(len(full_train_dataset), SUBSET_SIZE)))
    val_indices = list(range(min(len(full_val_dataset), SUBSET_SIZE)))

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    print(f"Train subset size: {len(train_dataset)}")
    print(f"Val subset size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for demo
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Verify a batch
    images, targets, rec_ids = next(iter(train_loader))
    assert images.shape == (
        DEMO_BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Unexpected image shape: {images.shape}"
    assert targets.shape == (
        DEMO_BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Unexpected target shape: {targets.shape}"
    print("DataLoader verification passed.")

    # 3. Model Initialization
    print("\n--- Step 3: Model Initialization ---")
    device = Config.DEVICE
    model = get_resnet34(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Verify forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)
        output = model(dummy_input)
        assert output.shape == (
            2,
            Config.NUM_CLASSES,
        ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass verified.")

    # 4. Training Loop
    print("\n--- Step 4: Training Demo ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        swa_start_epoch=DEMO_SWA_START,  # Start SWA immediately for demo
        logger=logger,
    )

    print(f"Running training for {DEMO_EPOCHS} epochs...")
    best_auc, swa_auc = trainer.fit(
        num_epochs=DEMO_EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        checkpoint_prefix="demo_model",
        patience=5,
    )

    # Verify checkpoints exist
    expected_checkpoints = [
        "demo_model_last.pth",
        "demo_model_base_best.pth",
        "demo_model_swa.pth",
    ]
    for ckpt in expected_checkpoints:
        path = os.path.join(Config.CHECKPOINT_DIR, ckpt)
        assert os.path.exists(path), f"Checkpoint not found: {path}"
    print("Training complete and checkpoints verified.")

    # 5. Inference: Pseudo-Label Generation (Teacher Mode)
    print("\n--- Step 5: Pseudo-Label Generation (Teacher) ---")
    # We use the SWA model trained above as the 'teacher'
    teacher_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "demo_model_swa.pth")

    # Force regeneration by ignoring cache if it exists (for demo purposes)
    if os.path.exists(Config.PSEUDO_LABEL_PATH):
        os.remove(Config.PSEUDO_LABEL_PATH)

    df_pseudo = generate_calibrated_pseudo_labels(
        teacher_checkpoint_paths=[teacher_ckpt_path],
        device=device,
        load_cached_data=False,
    )

    # Verify pseudo labels
    assert isinstance(df_pseudo, pd.DataFrame)
    assert "rec_id" in df_pseudo.columns
    # Check for species columns
    expected_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    for col in expected_cols:
        assert col in df_pseudo.columns

    # Test set has 64 samples
    assert len(df_pseudo) == 64, f"Expected 64 pseudo-labels, got {len(df_pseudo)}"
    print("Pseudo-labels generated and verified.")

    # 6. Inference: Submission Generation (Student Mode)
    print("\n--- Step 6: Submission Generation (Student) ---")
    # We use the base best model as the 'student' for this demo
    student_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "demo_model_base_best.pth")

    predict_student(student_checkpoint_path=student_ckpt_path, device=device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "Id" in df_sub.columns
    assert "Probability" in df_sub.columns

    # Expected rows = 64 test samples * 19 classes = 1216
    expected_rows = 64 * 19
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Verify Id format (rec_id * 100 + species_id)
    # Check first Id
    first_id = df_sub.iloc[0]["Id"]
    # The test set rec_ids start from various numbers, but let's just check type
    assert isinstance(first_id, (int, np.integer))

    print("Submission file generated and verified.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
