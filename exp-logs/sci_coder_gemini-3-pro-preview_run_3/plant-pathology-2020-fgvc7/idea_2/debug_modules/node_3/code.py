import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library modules
import library.config
import library.data
import library.models
import library.training
import library.inference
import library.utils


def main():
    print("==== Starting Apple Disease Detection Pipeline Demonstration ====")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Patching configuration for fast execution...")

    # We patch these values to ensure the demo finishes quickly (1 epoch, small batch)
    # Since modules import these variables using 'from ... import ...', we must patch
    # them in the specific modules where they are used.

    NEW_EPOCHS = 1
    NEW_BATCH_SIZE = 4
    NEW_PATIENCE = 1

    # Patch library.config (source of truth)
    library.config.EPOCHS = NEW_EPOCHS
    library.config.BATCH_SIZE = NEW_BATCH_SIZE
    library.config.PATIENCE = NEW_PATIENCE

    # Patch library.training (uses EPOCHS, BATCH_SIZE, PATIENCE)
    library.training.EPOCHS = NEW_EPOCHS
    library.training.BATCH_SIZE = NEW_BATCH_SIZE
    library.training.PATIENCE = NEW_PATIENCE

    # Patch library.data (uses BATCH_SIZE)
    library.data.BATCH_SIZE = NEW_BATCH_SIZE

    # Patch library.inference (uses BATCH_SIZE)
    library.inference.BATCH_SIZE = NEW_BATCH_SIZE

    print(f"    EPOCHS set to {library.training.EPOCHS}")
    print(f"    BATCH_SIZE set to {library.data.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Get dataloaders in debug mode (subsamples data)
    train_loader, val_loader, test_loader = library.data.get_dataloaders(
        debug=True,
        batch_size=NEW_BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"    Train Batch Shape: Images={images.shape}, Labels={labels.shape}")

    # Assertions
    assert images.shape[0] == NEW_BATCH_SIZE, "Train batch size mismatch"
    assert images.shape[1] == 3, "Image channel mismatch (should be 3)"
    assert images.shape[2] == library.config.IMG_SIZE, "Image height mismatch"
    assert images.shape[3] == library.config.IMG_SIZE, "Image width mismatch"
    assert labels.shape[0] == NEW_BATCH_SIZE, "Label batch size mismatch"

    # Verify Test Loader (returns images and IDs)
    test_images, test_ids = next(iter(test_loader))
    print(f"    Test Batch Shape: Images={test_images.shape}, IDs={len(test_ids)}")
    assert len(test_ids) == NEW_BATCH_SIZE, "Test ID batch size mismatch"

    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model_name = library.config.MODEL_CONVNEXT
    num_classes = library.config.NUM_CLASSES

    # Instantiate model
    model = library.models.get_model(
        model_name, pretrained=False, num_classes=num_classes
    )
    model.eval()

    # Forward pass with dummy data
    dummy_input = torch.randn(2, 3, library.config.IMG_SIZE, library.config.IMG_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        num_classes,
    ), f"Model output shape mismatch. Expected (2, {num_classes})"
    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Class Weights Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Class Weights Calculation...")

    train_df = pd.read_csv(library.config.TRAIN_CSV)
    weights = library.utils.calculate_class_weights(train_df, device="cpu")

    print(f"    Class Weights: {weights}")
    assert (
        weights.shape[0] == library.config.NUM_CLASSES
    ), "Class weights dimension mismatch"
    assert torch.is_tensor(weights), "Class weights should be a Tensor"
    print("    Class Weights verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Integration (Train 2 models briefly)
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Integration (Debug Mode)...")

    # Define paths for saved models
    convnext_save_name = "convnext_debug.pth"
    effnet_save_name = "effnet_debug.pth"

    # Train ConvNeXt
    print("    -> Training ConvNeXt...")
    model_conv, hist_conv = library.training.run_training(
        model_name=library.config.MODEL_CONVNEXT,
        use_swa=False,
        debug=True,
        save_name=convnext_save_name,
    )

    # Train EfficientNet
    print("    -> Training EfficientNet...")
    model_eff, hist_eff = library.training.run_training(
        model_name=library.config.MODEL_EFFICIENTNET,
        use_swa=False,
        debug=True,
        save_name=effnet_save_name,
    )

    # Verify outputs
    convnext_path = os.path.join(library.config.CACHE_DIR, convnext_save_name)
    effnet_path = os.path.join(library.config.CACHE_DIR, effnet_save_name)

    assert os.path.exists(convnext_path), "ConvNeXt model file not saved"
    assert os.path.exists(effnet_path), "EfficientNet model file not saved"
    assert "train_loss" in hist_conv, "Training history missing train_loss"

    print("    Training integration passed.")

    # -------------------------------------------------------------------------
    # 6. Inference & Ensemble Integration
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference & Ensemble Integration...")

    # Run ensemble prediction using the models we just trained
    library.inference.ensemble_predictions(
        effnet_weights_path=effnet_path, convnext_weights_path=convnext_path, debug=True
    )

    # Verify submission file
    submission_path = library.config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Assertions
    expected_cols = ["image_id"] + library.config.CLASS_LABELS
    for col in expected_cols:
        assert col in df_sub.columns, f"Missing column in submission: {col}"

    # Since we ran in debug mode, the number of rows depends on the debug subsampling in get_dataloaders.
    # get_dataloaders(debug=True) takes head(batch_size).
    # So we expect NEW_BATCH_SIZE rows.
    assert (
        len(df_sub) == NEW_BATCH_SIZE
    ), f"Submission row count mismatch. Expected {NEW_BATCH_SIZE}, got {len(df_sub)}"

    print("    Inference integration passed.")

    print("\n==== All Demonstration Steps Completed Successfully ====")


if __name__ == "__main__":
    # Ensure reproducibility
    library.config.seed_everything(42)
    main()
