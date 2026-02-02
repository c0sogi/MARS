import os
import sys
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_setup import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.trainer import Trainer
from library.inference import predict_test_set

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")
    # Override Config attributes to ensure the script runs quickly and fits in memory
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.IMAGE_SIZE = 128  # Reduce image size for faster processing
    Config.DEBUG = True  # Enable debug mode

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Image Size: {Config.IMAGE_SIZE}")
    print(f"  Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading (Train/Val)
    # ---------------------------------------------------------
    print("\n[Step 2] Loading DataLoaders (Debug Subset)...")

    # Load a small subset of data (e.g., 50 samples)
    train_loader, val_loader, counts = get_dataloaders(
        debug=True,
        data_subset_size=50,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Validation: Check counts dictionary
    print("  Taxonomy Counts:", counts)
    assert "num_families" in counts
    assert "num_genera" in counts
    assert "num_species" in counts

    # Validation: Check Batch Structure
    print("  Verifying batch structure...")
    images, targets = next(iter(train_loader))

    # Check Image Shape: [Batch, 3, H, W]
    assert images.dim() == 4
    assert images.shape[0] == Config.BATCH_SIZE
    assert images.shape[1] == 3
    assert images.shape[2] == Config.IMAGE_SIZE
    assert images.shape[3] == Config.IMAGE_SIZE

    # Check Targets: Tuple of (species, genus, family)
    assert isinstance(targets, (list, tuple))
    assert len(targets) == 3
    species_lbl, genus_lbl, family_lbl = targets
    assert species_lbl.shape[0] == Config.BATCH_SIZE

    print("  Data Loading verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing HierarchicalEfficientNet...")

    model = HierarchicalEfficientNet(
        num_families=counts["num_families"],
        num_genera=counts["num_genera"],
        num_species=counts["num_species"],
        pretrained=False,  # Disable pretrained weights download for speed/offline safety
    )

    model.to(Config.DEVICE)

    # Validation: Forward Pass
    print("  Running dummy forward pass...")
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        outputs = model(dummy_input)

    # Check Outputs
    assert isinstance(outputs, dict)
    assert "species" in outputs
    assert "genus" in outputs
    assert "family" in outputs

    # Check Logit Shapes
    assert outputs["species"].shape == (Config.BATCH_SIZE, counts["num_species"])
    assert outputs["genus"].shape == (Config.BATCH_SIZE, counts["num_genera"])
    assert outputs["family"].shape == (Config.BATCH_SIZE, counts["num_families"])

    print("  Model architecture verified successfully.")

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, device=Config.DEVICE)

    # Run for just 1 epoch to demonstrate functionality
    trainer.fit(num_epochs=1)

    # Validation: Checkpoint creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created."
    print(f"  Checkpoint verified at: {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 5. Inference
    # ---------------------------------------------------------
    print("\n[Step 5] Running Inference on Test Subset...")

    # Create a small temporary test csv to avoid processing the full 170k+ test set
    full_test_df = pd.read_csv(Config.TEST_CSV)
    small_test_df = full_test_df.head(20).copy()  # Take top 20

    temp_test_csv = os.path.join(Config.WORKING_DIR, "temp_test_subset.csv")
    small_test_df.to_csv(temp_test_csv, index=False)

    # Override Config.TEST_CSV to point to our temporary file
    # Note: The library code uses Config.TEST_CSV as default in get_test_dataloader
    # We must patch the Config class attribute
    original_test_csv_path = Config.TEST_CSV
    Config.TEST_CSV = temp_test_csv

    try:
        # Run inference
        # This function loads the model from Config.MODEL_SAVE_PATH (created in Step 4)
        predict_test_set(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            device=Config.DEVICE,
            use_tta=False,
            load_cached_taxonomy=True,
        )

        # Validation: Submission file
        assert os.path.exists(
            Config.SUBMISSION_PATH
        ), "Submission file was not created."

        submission_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission loaded. Shape: {submission_df.shape}")

        # Check rows match our small test set
        assert len(submission_df) == len(small_test_df)
        assert "Id" in submission_df.columns
        assert "Predicted" in submission_df.columns

        print("  Inference verified successfully.")

    finally:
        # Restore Config (good practice, though script ends here)
        Config.TEST_CSV = original_test_csv_path

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)
    run_demo()
