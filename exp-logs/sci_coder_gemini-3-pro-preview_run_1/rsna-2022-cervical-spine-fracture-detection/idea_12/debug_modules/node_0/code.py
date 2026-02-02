import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import setup_logger, calculate_weighted_log_loss
from library.models import UNetLocalizer, DualStreamEncoder, SpinalGraphAggregator
from library.trainer import Trainer
from library.inference import InferencePipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting RSNA Fracture Detection Demo ===")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration and environment...")

    # Set seed for reproducibility
    seed_everything(42)

    # Define a demo working directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.LOG_DIR = os.path.join(DEMO_DIR, "logs")

    # Re-run setup to create these new directories
    Config.setup()

    # Override Hyperparameters for speed
    Config.BATCH_SIZE_SEG = 2
    Config.BATCH_SIZE_CLS = 4
    Config.BATCH_SIZE_SEQ = 2
    Config.EPOCHS_SEG = 1
    Config.EPOCHS_CLS = 1
    Config.EPOCHS_SEQ = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    # ---------------------------------------------------------
    # 2. Create Mini Datasets (Subsetting)
    # ---------------------------------------------------------
    print("\n[2] Creating mini datasets for demonstration...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Filter train set to include some samples WITH segmentation for Stage 1
    # and some random ones for diversity
    seg_samples = orig_train[orig_train["has_segmentation"]].head(3)
    non_seg_samples = orig_train[~orig_train["has_segmentation"]].head(2)
    mini_train = pd.concat([seg_samples, non_seg_samples]).reset_index(drop=True)

    # Mini Val
    mini_val = orig_val.head(5).reset_index(drop=True)

    # Mini Test
    mini_test = orig_test.head(3).reset_index(drop=True)

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"    Train subset: {len(mini_train)} rows")
    print(f"    Val subset:   {len(mini_val)} rows")
    print(f"    Test subset:  {len(mini_test)} rows")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architectures...")
    device = Config.DEVICE

    # A. UNet Localizer
    print("    Verifying UNetLocalizer...")
    model_seg = UNetLocalizer(n_classes=8).to(device)
    dummy_input_seg = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out_seg = model_seg(dummy_input_seg)
    assert out_seg.shape == (
        2,
        8,
        256,
        256,
    ), f"UNet output shape mismatch: {out_seg.shape}"
    print("    -> UNet Passed.")

    # B. Dual Stream Encoder
    print("    Verifying DualStreamEncoder...")
    model_enc = DualStreamEncoder(feature_dim=1280).to(device)
    dummy_local = torch.randn(2, 2, 256, 256).to(device)
    dummy_global = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out_enc = model_enc(dummy_local, dummy_global)
    # The model returns projected features (Batch, 1280)
    assert out_enc.shape == (2, 1280), f"Encoder output shape mismatch: {out_enc.shape}"
    print("    -> Encoder Passed.")

    # C. Spinal Graph Aggregator
    print("    Verifying SpinalGraphAggregator...")
    model_agg = SpinalGraphAggregator(input_dim=1280, hidden_dim=256, gcn_dim=128).to(
        device
    )
    seq_len = 10
    dummy_seq = torch.randn(2, seq_len, 1280).to(device)
    dummy_probs = torch.randn(2, seq_len, 8).to(device)  # Anatomical probs
    with torch.no_grad():
        vert_probs, patient_prob = model_agg(dummy_seq, dummy_probs)
    assert vert_probs.shape == (
        2,
        7,
    ), f"Vertebrae probs shape mismatch: {vert_probs.shape}"
    assert patient_prob.shape == (
        2,
        1,
    ), f"Patient prob shape mismatch: {patient_prob.shape}"
    print("    -> Aggregator Passed.")

    # ---------------------------------------------------------
    # 4. Training Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n[4] Running Training Pipeline...")
    trainer = Trainer()

    # Stage 1: Localizer
    print("    -> Training Localizer (Stage 1)...")
    trainer.train_localizer(epochs=Config.EPOCHS_SEG)
    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
    ), "Stage 1 checkpoint missing"

    # Stage 2: Encoder
    print("    -> Training Encoder (Stage 2)...")
    trainer.train_encoder(epochs=Config.EPOCHS_CLS)
    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
    ), "Stage 2 checkpoint missing"

    # Feature Extraction
    print("    -> Extracting Features...")
    trainer.extract_features(load_cached_data=False)
    # Verify features were created
    feature_files = os.listdir(os.path.join(Config.CACHE_DIR, "features"))
    assert len(feature_files) > 0, "No features extracted"
    print(f"       Extracted features for {len(feature_files)} studies.")

    # Stage 3: Aggregator
    print("    -> Training Aggregator (Stage 3)...")
    trainer.train_aggregator(epochs=Config.EPOCHS_SEQ)
    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    ), "Stage 3 checkpoint missing"

    # ---------------------------------------------------------
    # 5. Inference Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Inference Pipeline...")
    pipeline = InferencePipeline()
    pipeline.run_inference()

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file not created"

    sub_df = pd.read_csv(sub_path)
    print(f"    Submission created with {len(sub_df)} rows.")
    print("    Sample predictions:")
    print(sub_df.head())

    # Basic check on submission format
    assert "row_id" in sub_df.columns and "fractured" in sub_df.columns
    assert (
        len(sub_df) == len(mini_test) * 8
    ), f"Expected {len(mini_test)*8} rows, got {len(sub_df)}"

    # ---------------------------------------------------------
    # 6. Metric Validation
    # ---------------------------------------------------------
    print("\n[6] Validating Metric Function...")
    # Create dummy true labels and predictions
    # Shape: (N, 8) -> [patient, C1..C7]
    y_true = np.array([[1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    # Perfect prediction
    y_pred_perfect = np.array(
        [
            [0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        ]
    )
    # Terrible prediction
    y_pred_bad = 1.0 - y_pred_perfect

    loss_perfect = calculate_weighted_log_loss(y_true, y_pred_perfect)
    loss_bad = calculate_weighted_log_loss(y_true, y_pred_bad)

    print(f"    Loss (Good Preds): {loss_perfect:.6f}")
    print(f"    Loss (Bad Preds):  {loss_bad:.6f}")

    assert (
        loss_perfect < loss_bad
    ), "Metric logic error: Good predictions should have lower loss."
    print("    -> Metric Validated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
