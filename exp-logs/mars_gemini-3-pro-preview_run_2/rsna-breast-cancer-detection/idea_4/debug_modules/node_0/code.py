import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders, BreastCancerBagDataset
from library.model import BreastCancerMILModel
from library.train import run_training


def main():
    print("Initializing Demonstration Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config defaults for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 bags per split
    Config.IMG_SIZE = (256, 256)  # Reduce resolution from 640 to 256
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure working directory is clean for fresh cache generation
    if os.path.exists(Config.WORKING_DIR):
        # We keep the directory but might want to clear old parquet caches if they exist
        # to ensure the DEBUG sampling logic runs fresh.
        for f in os.listdir(Config.WORKING_DIR):
            if f.endswith(".parquet"):
                os.remove(os.path.join(Config.WORKING_DIR, f))

    # Run setup (creates directories, sets seeds)
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print("    Configuration updated successfully.")

    # ==========================================
    # 2. Metric Verification
    # ==========================================
    print("\n[2] Verifying Probabilistic F1 Score (pF1)...")

    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Case 2: Worst prediction
    y_pred_worst = np.array([0.0, 1.0, 0.0, 1.0])
    score_worst = probabilistic_f1(y_true, y_pred_worst)
    assert np.isclose(score_worst, 0.0), f"Expected 0.0, got {score_worst}"

    # Case 3: Probabilistic inputs
    y_pred_prob = np.array([0.8, 0.2, 0.6, 0.1])
    # pTP = 0.8*1 + 0.2*0 + 0.6*1 + 0.1*0 = 1.4
    # pFP = 0.8*0 + 0.2*1 + 0.6*0 + 0.1*1 = 0.3
    # Total Positives = 2
    # pPrecision = 1.4 / (1.4 + 0.3) = 1.4 / 1.7 ≈ 0.8235
    # pRecall = 1.4 / 2 = 0.7
    # pF1 = 2 * (0.8235 * 0.7) / (0.8235 + 0.7) ≈ 0.7567
    score_prob = probabilistic_f1(y_true, y_pred_prob)
    assert 0.7 < score_prob < 0.8, f"pF1 calculation seems off: {score_prob}"

    print(f"    pF1 Logic Verified. Score for probabilistic inputs: {score_prob:.4f}")

    # ==========================================
    # 3. Data Loading & Batch Structure
    # ==========================================
    print("\n[3] Verifying DataLoaders and Batch Structure...")

    # Force reload to apply DEBUG settings
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["images"]  # (B, Max_V, C, H, W)
    mask = batch["mask"]  # (B, Max_V)
    metadata = batch["metadata"]  # (B, 3)
    labels = batch["labels"]  # (B, 1)

    # Assertions
    assert images.dim() == 5, f"Expected 5D images tensor, got {images.shape}"
    assert mask.dim() == 2, f"Expected 2D mask tensor, got {mask.shape}"
    assert (
        metadata.shape[1] == 3
    ), f"Expected 3 metadata features, got {metadata.shape[1]}"
    assert labels.shape[1] == 1, f"Expected 1 label column, got {labels.shape[1]}"
    assert images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    print(f"    Batch Shapes Verified:")
    print(f"      Images:   {images.shape}")
    print(f"      Mask:     {mask.shape}")
    print(f"      Metadata: {metadata.shape}")
    print(f"      Labels:   {labels.shape}")

    # ==========================================
    # 4. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    model = BreastCancerMILModel(config=Config)
    model.to(device)

    # Move batch to device
    images = images.to(device)
    mask = mask.to(device)
    metadata = metadata.to(device)
    labels = labels.to(device)

    # Forward Pass
    logits = model(images, mask, metadata)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {logits.shape}"

    # Backward Pass Check (Gradient flow)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    loss.backward()

    # Check if gradients are populated for a key layer
    assert (
        model.classifier[0].weight.grad is not None
    ), "Gradients not computed for classifier."

    print("    Model Forward/Backward pass successful.")
    print(f"    Initial Loss: {loss.item():.4f}")

    # ==========================================
    # 5. Full Training Pipeline Integration
    # ==========================================
    print("\n[5] Executing Training Pipeline (1 Epoch)...")

    # This calls the provided training script logic
    trained_model = run_training()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("    Training pipeline completed successfully.")

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    print("\n[6] Generating Submission...")

    trained_model.eval()
    predictions = []
    prediction_ids = []

    print("    Running inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["images"].to(device)
            msk = batch["mask"].to(device)
            meta = batch["metadata"].to(device)
            ids = batch["ids"]

            # Forward
            logits = trained_model(imgs, msk, meta)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            prediction_ids.extend(ids)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"prediction_id": prediction_ids, "cancer": predictions}
    )

    # Verify against sample submission
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Note: Since we are in DEBUG mode with a subset, the rows won't match exactly
    # with the full sample submission. However, we check the structure.
    assert "prediction_id" in submission_df.columns
    assert "cancer" in submission_df.columns
    assert submission_df["cancer"].min() >= 0.0 and submission_df["cancer"].max() <= 1.0

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Generated {len(submission_df)} predictions.")
    print("    Head of submission:")
    print(submission_df.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
