import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import seed_everything, AverageMeter, get_logger
from library.dataset import get_dataloaders
from library.model import HierarchicalMetricNet
from library.loss import HierarchicalMultiTaskLoss
from library.train import Trainer
from library.inference import predict_and_submit


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[Demo] 1. Configuring environment for fast demonstration...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Very small subset for quick execution
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.MODEL_NAME = "tf_efficientnet_b0_ns"  # Use smaller backbone
    Config.EMBEDDING_DIM = 64  # Smaller embedding size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.GRAD_ACCUM_STEPS = 1

    # Ensure working directories exist
    if os.path.exists(Config.WORKING_DIR):
        # clean up previous runs to ensure assertions are valid
        try:
            shutil.rmtree(Config.CACHE_DIR)
            if os.path.exists(Config.SUBMISSION_FILE):
                os.remove(Config.SUBMISSION_FILE)
            if os.path.exists(Config.BEST_MODEL_PATH):
                os.remove(Config.BEST_MODEL_PATH)
        except Exception:
            pass

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    logger = get_logger("demo_script")
    logger.info("Configuration updated for demo.")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n[Demo] 2. Verifying Utilities...")
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Total sum = 10*2 + 20*2 = 60. Total count = 4. Avg = 15.
    assert (
        meter.avg == 15.0
    ), f"AverageMeter logic failed. Expected 15.0, got {meter.avg}"
    print("  AverageMeter verified.")

    # ==========================================
    # 3. Verify Dataset & Dataloaders
    # ==========================================
    print("\n[Demo] 3. Verifying Dataset and DataLoaders...")
    # This will also trigger taxonomy processing and caching
    train_loader, val_loader, test_loader, meta_counts = get_dataloaders(
        debug=True, load_cached_data=False
    )

    assert len(train_loader) > 0, "Train loader is empty."
    assert meta_counts["num_species"] > 0, "No species found in metadata."

    # Fetch a batch
    images, species_lbl, genus_lbl, family_lbl = next(iter(train_loader))

    print(f"  Batch shapes: Images {images.shape}, Species {species_lbl.shape}")
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image batch shape."
    assert species_lbl.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape."
    print("  DataLoaders verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[Demo] 4. Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = HierarchicalMetricNet(
        num_species=meta_counts["num_species"],
        num_genera=meta_counts["num_genera"],
        num_families=meta_counts["num_families"],
    ).to(device)

    # Move batch to device
    images = images.to(device)
    species_lbl = species_lbl.to(device)

    # Forward pass (Training mode with labels)
    outputs = model(images, species_label=species_lbl)

    # Check output keys
    expected_keys = ["species", "genus", "family", "embedding"]
    for k in expected_keys:
        assert k in outputs, f"Model output missing key: {k}"

    # Check output shapes
    assert outputs["species"].shape == (
        Config.BATCH_SIZE,
        meta_counts["num_species"],
    ), "Species logits shape mismatch."
    assert outputs["genus"].shape == (
        Config.BATCH_SIZE,
        meta_counts["num_genera"],
    ), "Genus logits shape mismatch."
    assert outputs["embedding"].shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_DIM,
    ), "Embedding shape mismatch."

    print("  Model forward pass verified.")

    # ==========================================
    # 5. Verify Loss Function
    # ==========================================
    print("\n[Demo] 5. Verifying Loss Function...")
    criterion = HierarchicalMultiTaskLoss().to(device)

    # Prepare targets
    targets = (species_lbl, genus_lbl.to(device), family_lbl.to(device))

    loss, metrics = criterion(outputs, targets)

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."
    assert "loss_species" in metrics, "Metrics missing component losses."

    print(f"  Loss calculated: {loss.item():.4f}")
    print("  Loss function verified.")

    # ==========================================
    # 6. Verify Training Loop
    # ==========================================
    print("\n[Demo] 6. Verifying Training Loop (Trainer)...")
    # Initialize Trainer with debug=True
    trainer = Trainer(debug=True)

    # Run fit (1 epoch, tiny dataset)
    trainer.fit()

    # Check if model checkpoint was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("  Training loop completed and model saved.")

    # ==========================================
    # 7. Verify Inference & Submission
    # ==========================================
    print("\n[Demo] 7. Verifying Inference Pipeline...")

    # Run inference using the model we just trained
    predict_and_submit(
        checkpoint_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_FILE,
        debug=True,
    )

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check if Id is int (as required by competition format, though loaded as whatever pandas infers)
    # The provided sample submission had Id as int.
    assert pd.api.types.is_numeric_dtype(df_sub["Id"]), "Id column should be numeric."
    assert pd.api.types.is_numeric_dtype(
        df_sub["Predicted"]
    ), "Predicted column should be numeric."

    print("  Inference completed and submission verified.")
    print("\n[Demo] All verification steps passed successfully!")


if __name__ == "__main__":
    run_demo()
