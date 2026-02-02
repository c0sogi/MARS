import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import cv2
from library.config import Config
from library.utils import seed_everything, mean_average_precision
from library.dataset import HotelDataset, get_dataloaders, get_transforms
from library.model import HotelIdModel
from library.loss import SubCenterArcFaceLoss
from library.engine import train_model
from library.inference import run_inference


# =============================================================================
# 1. Setup and Configuration Patching
# =============================================================================
def setup_demo_config():
    """
    Patches the Config class to use a lightweight setting for demonstration.
    """
    print("[Setup] Patching Configuration for Demo...")

    # Create a specific working directory for this demo
    demo_working_dir = "./working/demo_run"
    os.makedirs(demo_working_dir, exist_ok=True)

    # Patch Config attributes
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_working_dir
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Use a lightweight model for speed
    Config.MODEL_NAMES = ["resnet18"]

    # Reduce training parameters
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.IMAGE_SIZE = 224  # Smaller image size for speed

    # We will determine NUM_CLASSES dynamically after subsetting data
    Config.SEED = 42

    # Update paths to point to the mini-metadata we are about to create
    Config.TRAIN_METADATA_PATH = os.path.join(
        demo_working_dir, "train_metadata_mini.csv"
    )
    Config.VAL_METADATA_PATH = os.path.join(demo_working_dir, "val_metadata_mini.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_working_dir, "test_metadata_mini.csv")

    # Reduce Inference parameters
    Config.DBA_K = 2
    Config.QE_K = 2
    Config.TOP_K = 5

    # Ensure clean state for caching
    if os.path.exists(os.path.join(Config.WORKING_DIR, "class_mapping.parquet")):
        os.remove(os.path.join(Config.WORKING_DIR, "class_mapping.parquet"))

    seed_everything(Config.SEED)
    print(f"[Setup] Working Directory: {Config.WORKING_DIR}")


# =============================================================================
# 2. Data Subsetting
# =============================================================================
def create_mini_dataset():
    """
    Creates a small subset of the original metadata for rapid testing.
    """
    print("[Data] Creating Mini Dataset...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Select top 5 classes with enough samples
    top_classes = orig_train["hotel_id"].value_counts().head(5).index.tolist()

    # Filter train and val
    mini_train = (
        orig_train[orig_train["hotel_id"].isin(top_classes)]
        .groupby("hotel_id")
        .head(10)
        .reset_index(drop=True)
    )
    mini_val = (
        orig_val[orig_val["hotel_id"].isin(top_classes)]
        .groupby("hotel_id")
        .head(5)
        .reset_index(drop=True)
    )

    # For test, just take a random sample of 10 images
    mini_test = orig_test.sample(n=10, random_state=Config.SEED).reset_index(drop=True)

    # Update Config.NUM_CLASSES
    Config.NUM_CLASSES = len(top_classes)
    print(f"[Data] Selected {Config.NUM_CLASSES} classes for demo.")
    print(f"[Data] Mini Train Size: {len(mini_train)}")
    print(f"[Data] Mini Val Size: {len(mini_val)}")
    print(f"[Data] Mini Test Size: {len(mini_test)}")

    # Save to working directory
    mini_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    mini_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    mini_test.to_csv(Config.TEST_METADATA_PATH, index=False)


# =============================================================================
# 3. Component Verification
# =============================================================================
def verify_components():
    print("\n[Verification] Starting Component Verification...")

    # --- 3.1 Verify Metric ---
    print("  -> Verifying MAP@5 Metric...")
    # Case: Target is 1st prediction (Rank 1) -> Score 1.0
    # Case: Target is 2nd prediction (Rank 2) -> Score 0.5
    preds = [[10, 20, 30, 40, 50], [10, 20, 30, 40, 50]]
    targets = [10, 20]
    score = mean_average_precision(preds, targets, k=5)
    expected = (1.0 + 0.5) / 2.0
    assert np.isclose(
        score, expected
    ), f"MAP calculation incorrect. Got {score}, expected {expected}"
    print("     MAP@5 Logic Verified.")

    # --- 3.2 Verify Dataset ---
    print("  -> Verifying Dataset and Transforms...")
    # Load the mini metadata we just created
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Create a dummy mapping
    unique_ids = sorted(df["hotel_id"].unique())
    class_to_idx = {cid: i for i, cid in enumerate(unique_ids)}

    dataset = HotelDataset(
        df, transforms=get_transforms("train"), is_test=False, class_to_idx=class_to_idx
    )

    assert len(dataset) == len(df)
    img, label = dataset[0]

    # Check tensor shape (C, H, W)
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch. Got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    print(f"     Dataset Verified. Sample shape: {img.shape}")

    # --- 3.3 Verify Model ---
    print("  -> Verifying Model Architecture...")
    device = Config.DEVICE
    model = HotelIdModel(
        model_name=Config.MODEL_NAMES[0], embedding_size=Config.EMBEDDING_SIZE
    )
    model.to(device)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.EMBEDDING_SIZE,
    ), f"Model output shape mismatch. Got {output.shape}, expected (2, {Config.EMBEDDING_SIZE})"
    print("     Model Forward Pass Verified.")

    # --- 3.4 Verify Loss ---
    print("  -> Verifying SubCenterArcFaceLoss...")
    loss_fn = SubCenterArcFaceLoss(
        num_classes=Config.NUM_CLASSES, embedding_size=Config.EMBEDDING_SIZE
    ).to(device)

    dummy_labels = torch.tensor([0, 1]).to(device)
    loss = loss_fn(output, dummy_labels)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"     Loss Calculation Verified. Loss: {loss.item():.4f}")

    # Clean up memory
    del model, loss_fn, dummy_input, output
    torch.cuda.empty_cache()


# =============================================================================
# 4. Training Pipeline Demonstration
# =============================================================================
def demonstrate_training():
    print("\n[Demo] Starting Training Pipeline...")

    # Get dataloaders (this will generate the class mapping cache based on mini dataset)
    train_loader, val_loader, _, _, _ = get_dataloaders(load_cached_data=False)

    # Train the model (Config.EPOCHS is set to 1)
    # This uses library.engine.train_model
    best_model_path = train_model(
        model_name=Config.MODEL_NAMES[0],
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
    )

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"[Demo] Training completed. Model saved at {best_model_path}")


# =============================================================================
# 5. Inference Pipeline Demonstration
# =============================================================================
def demonstrate_inference():
    print("\n[Demo] Starting Inference Pipeline...")

    # Run the full inference pipeline provided in library.inference
    # This includes Feature Extraction, DBA, QE, and Prediction
    run_inference(load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Check submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"[Demo] Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    # Basic format check
    assert (
        "image" in df_sub.columns and "hotel_id" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission is empty."


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    try:
        # 1. Setup
        setup_demo_config()

        # 2. Data
        create_mini_dataset()

        # 3. Verify
        verify_components()

        # 4. Train
        demonstrate_training()

        # 5. Inference
        demonstrate_inference()

        print("\n[Success] All demonstrations and verifications passed successfully.")

    except Exception as e:
        print(f"\n[Error] An error occurred: {e}")
        # Print traceback for debugging if needed, but simple error message is often enough
        import traceback

        traceback.print_exc()
        sys.exit(1)
