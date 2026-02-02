import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, process_hierarchy_mappings
from library.dataset import get_loaders
from library.model import get_model
from library.train import fit
from library.inference import predict_all

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Demonstration
    # --------------------------------------------------------------------------
    print("==== Setting up configuration for demonstration ====")

    # Override Config defaults for a fast demo run
    Config.SEED = 42
    Config.DEBUG = True  # Enables sampling (2000 train, 500 val/test)
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.USE_TTA = False  # Disable TTA to speed up inference

    # Clean up and recreate the working directory for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # --------------------------------------------------------------------------
    # 2. Verify Hierarchy Mapping Logic
    # --------------------------------------------------------------------------
    print("\n==== Verifying Hierarchy Mapping ====")
    # This function parses train_metadata.json and creates mappings for multi-task learning
    hierarchy_df = process_hierarchy_mappings(
        Config.TRAIN_METADATA_JSON, Config.WORKING_DIR, load_cached_data=False
    )

    # Assertions to ensure mapping is correct
    assert not hierarchy_df.empty, "Hierarchy DataFrame should not be empty."
    assert "category_id" in hierarchy_df.columns, "Missing 'category_id' column."
    assert "genus_id" in hierarchy_df.columns, "Missing 'genus_id' column."
    assert "family_id" in hierarchy_df.columns, "Missing 'family_id' column."

    num_genera = hierarchy_df["genus_id"].max() + 1
    num_families = hierarchy_df["family_id"].max() + 1
    print(
        f"Hierarchy mapped successfully. Found {num_genera} genera and {num_families} families."
    )

    # --------------------------------------------------------------------------
    # 3. Verify Data Loaders
    # --------------------------------------------------------------------------
    print("\n==== Verifying Data Loaders ====")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Fetch one batch to verify shapes and types
    images, species_ids, genus_ids, family_ids = next(iter(train_loader))

    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Tensor Shape: {images.shape}")
    print(f"Species Labels Shape: {species_ids.shape}")

    # Assertions
    assert images.dim() == 4, "Images must be 4D tensors (N, C, H, W)."
    assert images.shape[1] == 3, "Images must have 3 channels (RGB)."
    assert (
        images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
    ), f"Images must be resized to {Config.IMG_SIZE}x{Config.IMG_SIZE}."
    assert species_ids.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch."
    assert genus_ids.shape[0] == Config.BATCH_SIZE, "Genus label batch size mismatch."
    assert family_ids.shape[0] == Config.BATCH_SIZE, "Family label batch size mismatch."
    print("Data loaders initialized and verified.")

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n==== Verifying Model Architecture ====")
    # Initialize model (pretrained backbone + 3 heads)
    model = get_model(pretrained=True, load_cached_hierarchy=True)
    model.eval()

    # Run a forward pass with the batch fetched earlier
    with torch.no_grad():
        outputs = model(images)

    # Check outputs
    assert "species" in outputs, "Model output missing 'species' head."
    assert "genus" in outputs, "Model output missing 'genus' head."
    assert "family" in outputs, "Model output missing 'family' head."

    # Verify output shapes: (Batch_Size, Num_Classes)
    assert outputs["species"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Species head output shape mismatch."
    assert outputs["genus"].shape == (
        Config.BATCH_SIZE,
        num_genera,
    ), "Genus head output shape mismatch."
    assert outputs["family"].shape == (
        Config.BATCH_SIZE,
        num_families,
    ), "Family head output shape mismatch."

    print(
        f"Model forward pass successful. Species logits shape: {outputs['species'].shape}"
    )

    # --------------------------------------------------------------------------
    # 5. Run Training (Fit)
    # --------------------------------------------------------------------------
    print("\n==== Running Training (1 Epoch, Debug Subset) ====")
    # fit() handles the training loop, validation, and saving the best model
    best_f1 = fit(epochs=Config.EPOCHS, load_cached_data=True)

    print(f"Training completed. Best Validation F1: {best_f1:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Model checkpoint not found at {checkpoint_path}"
    print("Checkpoint verified.")

    # --------------------------------------------------------------------------
    # 6. Run Inference
    # --------------------------------------------------------------------------
    print("\n==== Running Inference ====")
    # predict_all() loads the best model and generates submission.csv
    # We use debug=True here to match the Config.DEBUG setting used for loaders
    predict_all(load_cached_data=True, debug=True)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Load and check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {sub_df.shape}")
    print(sub_df.head())

    assert "Id" in sub_df.columns, "Submission missing 'Id' column."
    assert "Predicted" in sub_df.columns, "Submission missing 'Predicted' column."
    assert not sub_df.empty, "Submission file is empty."

    # Check if Id is integer (as per sample_submission)
    assert pd.api.types.is_integer_dtype(
        sub_df["Id"]
    ), "Id column should be integer type."

    print("\n==== Demonstration Completed Successfully ====")
