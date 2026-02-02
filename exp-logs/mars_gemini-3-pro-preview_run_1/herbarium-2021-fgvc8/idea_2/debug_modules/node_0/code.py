import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure library imports work
sys.path.append(".")

from library.utils import seed_everything
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import HerbariumEfficientNet
from library.trainer import Trainer
from library.inference import predict_and_submit


def run_demo():
    print("==== 1. Setup & Reproducibility ====")
    # Set fixed seed
    seed_everything(42)

    # Define constants for the demo
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKING_DIR = "./working/demo_run"
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = "./working/submission.csv"

    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # Create working directory
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("\n==== 2. Data Pipeline Verification ====")
    # Get dataloaders in debug mode (uses small subset of data)
    print("Initializing dataloaders (debug=True)...")
    train_loader, val_loader, num_classes = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampling_mode="balanced",  # Test sampler logic
        load_cached_data=False,  # Force re-computation for demo
        debug=True,  # Use subset
    )

    print(f"Number of classes in subset: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Train Batch
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (BATCH_SIZE, 3, 224, 224), "Incorrect image tensor shape"
    assert targets.shape == (BATCH_SIZE,), "Incorrect target tensor shape"
    assert isinstance(images, torch.Tensor), "Images should be a torch.Tensor"
    assert isinstance(targets, torch.Tensor), "Targets should be a torch.Tensor"

    print("\n==== 3. Model Logic Verification ====")
    model = HerbariumEfficientNet(num_classes=num_classes)
    model = model.to(DEVICE)

    # Test Forward Pass
    dummy_input = images.to(DEVICE)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (BATCH_SIZE, num_classes), "Model output shape mismatch"

    # Test Freezing Logic
    print("Testing backbone freezing...")
    model.freeze_backbone()

    # Check if backbone params are frozen (requires_grad=False)
    # and classifier params are active (requires_grad=True)
    # Note: In timm efficientnet, the head is usually 'classifier'
    frozen_params = 0
    active_params = 0
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            active_params += 1
            # Assert that active params belong to classifier
            # This check depends on specific model architecture naming,
            # but HerbariumEfficientNet logic specifically unfreezes 'classifier' or 'fc'
            assert (
                "classifier" in name or "fc" in name
            ), f"Parameter {name} should be frozen but is active."
        else:
            frozen_params += 1

    print(f"Frozen parameters: {frozen_params}, Active parameters: {active_params}")
    assert frozen_params > 0, "Backbone should be frozen"
    assert active_params > 0, "Classifier should be active"

    # Test Unfreezing
    print("Testing unfreezing...")
    model.unfreeze_all()
    all_active = all(p.requires_grad for p in model.parameters())
    assert all_active, "All parameters should be trainable after unfreeze_all()"
    print("Model logic verified.")

    print("\n==== 4. Training Loop Demonstration ====")
    # Setup Trainer components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        save_dir=WORKING_DIR,
    )

    print("Starting training (1 epoch, debug mode)...")
    # Train for 1 epoch
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        patience=1,
        checkpoint_name="model.pth",
    )

    # Verify checkpoint creation
    assert os.path.exists(MODEL_PATH), f"Model checkpoint not found at {MODEL_PATH}"
    print(f"Checkpoint successfully saved to {MODEL_PATH}")

    print("\n==== 5. Inference Demonstration ====")
    # We use the high-level inference function provided in library.inference
    # We point it to the model we just trained

    print("Running inference...")
    predict_and_submit(
        checkpoint_path=MODEL_PATH,
        output_file=SUBMISSION_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        debug=True,  # Use subset of test data
    )

    # Verify submission file
    assert os.path.exists(
        SUBMISSION_PATH
    ), f"Submission file not found at {SUBMISSION_PATH}"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    assert list(df_sub.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"
    assert (
        df_sub["Id"].dtype == "int64" or df_sub["Id"].dtype == "int32"
    ), "Id column should be integer"
    assert (
        df_sub["Predicted"].dtype == "int64" or df_sub["Predicted"].dtype == "int32"
    ), "Predicted column should be integer"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
