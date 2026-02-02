import os
import torch
import numpy as np
import pandas as pd
import shutil
from library import config, utils, augmentations, dataset, model, trainer


def run_demo():
    print("=== Starting Demonstration of Speech Command Recognition Pipeline ===")

    # 1. Setup and Reproducibility
    print("\n[Step 1] Setting up environment and seeds...")
    utils.set_seed(42)

    # Override configuration for speed
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 16  # Smaller batch size for demo speed

    # Ensure working directories exist (handled by config, but good to double check logic)
    if os.path.exists(config.WORK_DIR):
        print(f"Working directory {config.WORK_DIR} exists.")
    else:
        os.makedirs(config.WORK_DIR)
        print(f"Created working directory {config.WORK_DIR}.")

    # 2. Test Augmentations
    print("\n[Step 2] Verifying SpecAugment logic...")
    # Create a dummy spectrogram: (Channels=3, Freq=64, Time=101)
    # Time dimension depends on STFT settings, usually ~101 for 1s audio with hop=160
    dummy_spec = torch.randn(3, config.N_MELS, 101)
    spec_augment = augmentations.SpecAugment(freq_mask_param=5, time_mask_param=5)

    augmented_spec = spec_augment(dummy_spec)

    # Assertions
    assert (
        augmented_spec.shape == dummy_spec.shape
    ), f"Augmentation changed shape: {augmented_spec.shape} vs {dummy_spec.shape}"
    assert not torch.equal(
        augmented_spec, dummy_spec
    ), "Augmentation did not modify the input tensor (random chance is extremely low)."
    print("SpecAugment verification passed.")

    # 3. Test Dataset and DataLoader
    print("\n[Step 3] Verifying Dataset and DataLoader...")
    # We use the get_dataloaders function from the library
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Assertions
    # Expected: (Batch, 3, 64, Time)
    assert inputs.dim() == 4, "Input must be 4D (B, C, F, T)"
    assert inputs.size(0) == config.BATCH_SIZE, "Batch size mismatch"
    assert (
        inputs.size(1) == 3
    ), "Should have 3 channels for Multi-Resolution Spectrogram"
    assert inputs.size(2) == config.N_MELS, f"Freq dim should be {config.N_MELS}"
    assert targets.size(0) == config.BATCH_SIZE, "Target batch size mismatch"
    assert targets.max() < config.NUM_CLASSES, "Target indices out of bounds"
    print("DataLoader verification passed.")

    # 4. Test Model Architecture
    print("\n[Step 4] Verifying Model Architecture...")
    # Instantiate model
    net = model.MultiResResNetCRNN(num_classes=config.NUM_CLASSES, pretrained=False)
    net.eval()  # Set to eval for deterministic pass

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        outputs = net(inputs)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(config.BATCH_SIZE, config.NUM_CLASSES)}, got {outputs.shape}"
    print("Model architecture verification passed.")

    # 5. Test Training Loop (Trainer)
    print("\n[Step 5] Verifying Training Loop (Dry Run)...")

    # Initialize Trainer
    task_trainer = trainer.Trainer()

    # Modify internal config references if necessary, or just rely on the global config change done in Step 1.
    # We run fit() with max_batches to limit runtime.
    # This will run training for 1 epoch (limited to 5 batches) and validation (limited to 5 batches),
    # then generate predictions.

    try:
        task_trainer.fit(num_epochs=1, max_batches=5)
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    # 6. Verify Outputs
    print("\n[Step 6] Verifying Output Files...")

    # Check Model Checkpoint
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Checkpoint found at: {config.MODEL_SAVE_PATH}")
        # Verify we can load it
        checkpoint = torch.load(config.MODEL_SAVE_PATH)
        assert "model_state_dict" in checkpoint, "Checkpoint missing model_state_dict"
    else:
        raise FileNotFoundError(
            f"Model checkpoint not generated at {config.MODEL_SAVE_PATH}"
        )

    # Check Submission File
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Submission file found at: {config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")

        # Validate Submission Format
        assert list(df_sub.columns) == ["fname", "label"], "Submission columns mismatch"
        assert len(df_sub) > 0, "Submission file is empty"

        # Check against sample submission length (approximate check)
        sample_sub = pd.read_csv("./input/sample_submission.csv")
        # Note: The test loader might drop the last batch if configured, but usually we want full predictions.
        # The provided dataset.py uses drop_last=False for test_loader, so lengths should match roughly
        # (ignoring potential filtering in dataset.py if any, though dataset.py seems to load all test files).
        # Let's just check it's reasonably populated.
        assert len(df_sub) == len(
            sample_sub
        ), f"Submission row count mismatch. Expected {len(sample_sub)}, got {len(df_sub)}"

    else:
        raise FileNotFoundError(
            f"Submission file not generated at {config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
