import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_taxonomy_mapping, compute_class_priors
from library.dataset import get_dataloaders, HerbariumDataset
from library.model import HierarchicalConvNeXt
from library.train import run_training_pipeline
from library.inference import run_inference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Setting up Configuration for Demo...")

    # Define demo-specific paths
    DEMO_WORK_DIR = "./working/demo_execution"
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_WORK_DIR, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)

    # Override Config for speed
    # We use a very small batch size and sample size to run in seconds
    Config.override(
        WORK_DIR=DEMO_WORK_DIR,
        SUBMISSION_DIR=DEMO_SUBMISSION_DIR,
        CHECKPOINT_PATH=os.path.join(DEMO_WORK_DIR, "checkpoint.pth"),
        BEST_MODEL_PATH=os.path.join(DEMO_WORK_DIR, "best_model.pth"),
        SUBMISSION_PATH=os.path.join(DEMO_SUBMISSION_DIR, "submission.csv"),
        TAXONOMY_MAP_PATH=os.path.join(DEMO_WORK_DIR, "taxonomy_mappings.parquet"),
        # Hyperparameters for fast execution
        BATCH_SIZE=4,
        DEBUG_SAMPLE_SIZE=20,  # Only use 20 images
        STAGE1_EPOCHS=1,  # 1 Epoch for Stage 1
        STAGE2_EPOCHS=1,  # 1 Epoch for Stage 2
        NUM_WORKERS=2,  # Reduce workers for small data
    )

    Config.setup()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ==========================================
    # 2. Verify Utilities (Taxonomy & Priors)
    # ==========================================
    print("\n>>> Verifying Utility Functions...")

    # Test Taxonomy Mapping
    print("Loading taxonomy mapping...")
    taxonomy_df = load_taxonomy_mapping(load_cached_data=False)
    assert isinstance(
        taxonomy_df, pd.DataFrame
    ), "Taxonomy mapping should be a DataFrame"
    assert "category_id" in taxonomy_df.columns
    assert "family_id" in taxonomy_df.columns
    assert "order_id" in taxonomy_df.columns
    print(f"Taxonomy mapping loaded. Shape: {taxonomy_df.shape}")

    # Test Class Priors
    print("Computing class priors...")
    priors = compute_class_priors(load_cached_data=False)
    assert isinstance(priors, np.ndarray), "Priors should be a numpy array"
    assert (
        len(priors) == Config.NUM_CLASSES
    ), f"Priors length {len(priors)} mismatch with NUM_CLASSES {Config.NUM_CLASSES}"
    print("Class priors computed successfully.")

    # ==========================================
    # 3. Verify Dataset & DataLoader
    # ==========================================
    print("\n>>> Verifying Dataset and DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        stage=1, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Verify Image Tensor
    print(f"Image batch shape: {images.shape}")
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert (
        images.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {images.shape[0]}"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == Config.IMAGE_SIZE and images.shape[3] == Config.IMAGE_SIZE
    ), "Image size mismatch"

    # Verify Labels (Multi-task tuple)
    assert isinstance(
        labels, (list, tuple)
    ), "Labels should be a tuple/list for multi-task learning"
    assert len(labels) == 3, "Labels should contain (species, family, order)"

    species_labels, family_labels, order_labels = labels
    print(
        f"Labels extracted: Species {species_labels.shape}, Family {family_labels.shape}, Order {order_labels.shape}"
    )
    assert species_labels.shape[0] == Config.BATCH_SIZE

    print("DataLoaders verified successfully.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n>>> Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = HierarchicalConvNeXt(pretrained=False)  # False for speed/offline demo
    model.to(device)

    # Forward Pass
    dummy_input = images.to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert isinstance(outputs, dict), "Model output should be a dictionary"
    assert "species" in outputs
    assert "family" in outputs
    assert "order" in outputs

    print(f"Model Output Keys: {list(outputs.keys())}")
    print(f"Species Logits Shape: {outputs['species'].shape}")

    assert outputs["species"].shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert outputs["family"].shape[0] == Config.BATCH_SIZE
    assert outputs["order"].shape[0] == Config.BATCH_SIZE

    # Verify Freezing Logic
    print("Testing backbone freezing...")
    model.freeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad is False, "Backbone parameters should be frozen"

    model.unfreeze_backbone()
    # Check one param to see if it's unfrozen
    for param in model.backbone.parameters():
        if param.requires_grad:
            break
    else:
        raise AssertionError("Backbone parameters should be unfrozen")

    print("Model architecture verified successfully.")

    # ==========================================
    # 5. Verify Training Pipeline
    # ==========================================
    print("\n>>> Running Training Pipeline (Integration Test)...")
    # This runs the full training loop for 1 epoch on the tiny dataset
    # It tests the optimizer, loss calculation, scaler, and saving logic

    try:
        run_training_pipeline(debug_size=Config.DEBUG_SAMPLE_SIZE)
        print("Training pipeline executed successfully.")
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise e

    # Check if artifacts were created
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not saved at {Config.BEST_MODEL_PATH}")

    print(f"Artifact check passed: {Config.BEST_MODEL_PATH} exists.")

    # ==========================================
    # 6. Verify Inference
    # ==========================================
    print("\n>>> Running Inference Pipeline...")

    try:
        run_inference(debug_size=Config.DEBUG_SAMPLE_SIZE)
        print("Inference pipeline executed successfully.")
    except Exception as e:
        print(f"Inference pipeline failed: {e}")
        raise e

    # Check submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    print("\n>>> All demonstrations and verifications completed successfully!")


if __name__ == "__main__":
    main()
