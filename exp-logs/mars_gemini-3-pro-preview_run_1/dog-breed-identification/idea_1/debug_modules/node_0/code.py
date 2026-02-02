import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration Patching
    # ==========================================
    # Import config first and patch values to ensure fast execution.
    # Subsequent imports will pick up these modified values.
    import library.config

    print("Configuring environment for fast demonstration...")
    library.config.DEBUG = True
    library.config.DEBUG_SAMPLE_SIZE = 64  # Process only 64 images per set
    library.config.EPOCHS_HEAD = 1  # 1 Epoch for Head Adaptation
    library.config.EPOCHS_FINETUNE = 1  # 1 Epoch for Fine-Tuning
    library.config.BATCH_SIZE = 16  # Smaller batch size for debug
    library.config.NUM_WORKERS = 0  # Use main process to avoid overhead

    # Import remaining library modules after patching
    import library.utils as utils
    import library.dataset as dataset
    import library.model as model
    import library.trainer as trainer

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("Verifying utilities...")
    utils.seed_everything(42)

    # Test save_submission logic with dummy data
    dummy_sub_path = os.path.join(library.config.WORKING_DIR, "test_submission.csv")
    dummy_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    dummy_ids = ["id_001", "id_002"]
    dummy_classes = ["breed_A", "breed_B"]

    utils.save_submission(
        dummy_probs, dummy_ids, dummy_classes, submission_path=dummy_sub_path
    )

    assert os.path.exists(dummy_sub_path), "Submission file was not created."
    df_check = pd.read_csv(dummy_sub_path)
    assert df_check.shape == (2, 3), "Incorrect submission shape."
    print("Utils verified.")

    # ==========================================
    # 3. Verify Dataset & DataLoaders
    # ==========================================
    print("Verifying dataset processing...")
    # Force reload to ensure DEBUG settings are applied (ignoring any pre-existing cache)
    train_loader, val_loader, test_loader, classes = dataset.get_dataloaders(
        load_cached_data=False
    )

    # Verify Class Mapping
    assert len(classes) == 120, f"Expected 120 classes, found {len(classes)}"

    # Verify Batch Dimensions
    images, labels = next(iter(train_loader))
    assert images.shape == (16, 3, 224, 224), f"Unexpected batch shape: {images.shape}"
    assert labels.shape == (16,), f"Unexpected label shape: {labels.shape}"
    print(f"Dataset verified. Loaded {len(classes)} classes.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("Verifying model architecture...")
    # Instantiate model on CPU for structure verification
    net = model.get_model(
        num_classes=len(classes), pretrained=False, device=torch.device("cpu")
    )

    # Verify Forward Pass Output Shape
    dummy_input = torch.randn(2, 3, 224, 224)
    output = net(dummy_input)
    assert output.shape == (2, 120), f"Model output shape mismatch: {output.shape}"

    # Verify Freeze Logic
    model.freeze_backbone(net)
    assert net.conv1.weight.requires_grad is False, "Backbone (conv1) should be frozen."
    assert net.fc.weight.requires_grad is True, "Head (fc) should be trainable."

    # Verify Unfreeze Logic
    model.unfreeze_layer4_and_head(net)
    assert (
        net.conv1.weight.requires_grad is False
    ), "Backbone (conv1) should remain frozen."
    assert (
        net.layer4[0].conv1.weight.requires_grad is True
    ), "Layer4 should be trainable."
    assert net.fc.weight.requires_grad is True, "Head should be trainable."
    print("Model logic verified.")

    # ==========================================
    # 5. Run Training Pipeline
    # ==========================================
    print("Running training pipeline...")
    # This executes the two-phase training and inference using the patched config
    trainer.run_training(train_loader, val_loader, test_loader, classes)

    # ==========================================
    # 6. Final Artifact Validation
    # ==========================================
    submission_path = library.config.SUBMISSION_PATH
    model_path = library.config.MODEL_SAVE_PATH

    assert os.path.exists(
        submission_path
    ), f"Submission file missing at {submission_path}"
    assert os.path.exists(model_path), f"Model file missing at {model_path}"

    # Verify final submission content
    sub_df = pd.read_csv(submission_path)
    # In DEBUG mode, the test set is truncated to DEBUG_SAMPLE_SIZE
    expected_rows = library.config.DEBUG_SAMPLE_SIZE
    assert (
        len(sub_df) == expected_rows
    ), f"Submission rows mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
