import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config, seed_everything
from library.dataset import get_loaders
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import map_at_5


def run_demonstration():
    print("============================================================")
    print("      Whale Identification: Pipeline Demonstration          ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config attributes to run a fast debug cycle
    Config.debug = True
    Config.epochs = 1
    Config.debug_sample_size = 30  # Small subset for speed
    Config.img_size = 128  # Reduced resolution
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.num_workers = 0  # Disable multiprocessing for simple demo script
    Config.backbone = "resnet18"  # Use a lightweight backbone

    # Define a specific working directory for this run
    Config.working_dir = "./working/demo_run"
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Update file paths in Config to point to the demo working directory
    Config.cache_train_images = os.path.join(Config.working_dir, "demo_train.npy")
    Config.cache_val_images = os.path.join(Config.working_dir, "demo_val.npy")
    Config.cache_test_images = os.path.join(Config.working_dir, "demo_test.npy")
    Config.model_path = os.path.join(Config.working_dir, "demo_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "demo_submission.csv")

    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Debug Mode: {Config.debug}")

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Metric Logic (MAP@5)...")

    # Test Case: Perfect prediction
    ground_truth = ["w_A", "w_B"]
    preds_perfect = [
        ["w_A", "w_x", "w_y", "w_z", "w_k"],
        ["w_B", "w_x", "w_y", "w_z", "w_k"],
    ]
    score_perfect = map_at_5(preds_perfect, ground_truth)
    assert abs(score_perfect - 1.0) < 1e-6, f"Expected 1.0, got {score_perfect}"

    # Test Case: Correct label at rank 2 (index 1) -> Score 1/2
    ground_truth_2 = ["w_A"]
    preds_rank2 = [["w_wrong", "w_A", "w_b", "w_c", "w_d"]]
    score_rank2 = map_at_5(preds_rank2, ground_truth_2)
    assert abs(score_rank2 - 0.5) < 1e-6, f"Expected 0.5, got {score_rank2}"

    print("    MAP@5 logic verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[3] Loading DataLoaders...")

    # get_loaders handles loading metadata, caching images, and creating DataLoaders
    train_loader, gallery_loader, val_loader, test_loader, id2label = get_loaders(
        debug=Config.debug,
        load_cached_data=False,  # Force processing from raw images first time
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    print(f"    Train Batch Image Shape: {images.shape}")
    print(f"    Train Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.train_batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), "Image tensor shape mismatch."
    assert labels.shape[0] == Config.train_batch_size, "Label batch size mismatch."

    # -------------------------------------------------------------------------
    # 4. Model & Loss Initialization
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model and Loss...")

    device = Config.device
    model = WhaleModel(
        backbone_name=Config.backbone, embedding_size=512, pretrained=False
    )
    model.to(device)
    model.eval()

    # Verify Forward Pass
    with torch.no_grad():
        embeddings = model(images.to(device))

    print(f"    Embedding Output Shape: {embeddings.shape}")
    assert embeddings.shape == (
        Config.train_batch_size,
        512,
    ), "Embedding shape mismatch."

    # Verify Loss
    loss_fn = ArcFaceLoss(in_features=512, out_features=Config.n_classes)
    loss_fn.to(device)

    # Compute dummy loss
    loss = loss_fn(embeddings, labels.to(device))
    print(f"    Initial Loss Value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN."

    # -------------------------------------------------------------------------
    # 5. Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop...")

    trainer = Trainer(debug=True)
    trainer.fit()

    # Verify model checkpoint creation
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError("Model checkpoint was not saved after training.")
    print(f"    Model saved to: {Config.model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference Pipeline...")

    # generate_submission loads the saved model and computes predictions
    generate_submission(load_cached_data=True)

    # Verify Submission File
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.submission_path)
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    First few rows:\n{df_sub.head(2)}")

    # Validation Checks on Submission
    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Missing required columns."

    # Check prediction format (space separated strings)
    sample_pred = df_sub.iloc[0]["Id"]
    assert isinstance(sample_pred, str), "Id column should contain strings."
    pred_tokens = sample_pred.split()
    assert len(pred_tokens) <= 5, "Predictions should not exceed 5 labels."

    print("\n============================================================")
    print("      Demonstration Completed Successfully                  ")
    print("============================================================")


if __name__ == "__main__":
    run_demonstration()
