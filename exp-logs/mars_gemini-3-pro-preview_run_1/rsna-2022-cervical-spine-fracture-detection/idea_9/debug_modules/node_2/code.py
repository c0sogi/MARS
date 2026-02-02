import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import WeightedLogLoss
from library.models import (
    AnatomicalLocalizer,
    DualBranchEncoder,
    HierarchicalAggregator,
)
from library.engine import (
    Stage1Trainer,
    Stage2Trainer,
    Stage3Trainer,
    SubmissionGenerator,
)
from library.data import process_segmentation_data, process_classification_data


def run_demo():
    print("=== Starting Cervical Spine Fracture Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo Speed
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment...")
    Config.seed_everything()

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Override Hyperparameters for fast execution
    Config.SEG_EPOCHS = 1
    Config.ENC_EPOCHS = 1
    Config.RNN_EPOCHS = 1

    Config.SEG_BATCH_SIZE = 2
    Config.ENC_BATCH_SIZE = 2
    Config.RNN_BATCH_SIZE = 2

    # -------------------------------------------------------------------------
    # 2. Create Mini Datasets (Subsetting)
    # -------------------------------------------------------------------------
    print("[2/6] Creating mini datasets for rapid testing...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Create Mini Train: Ensure we have at least one sample with segmentation for Stage 1
    # and some without for general robustness
    seg_samples = orig_train[orig_train["has_segmentation"] == True].head(2)
    non_seg_samples = orig_train[orig_train["has_segmentation"] == False].head(2)
    mini_train = pd.concat([seg_samples, non_seg_samples])

    # Create Mini Val
    mini_val = orig_val.head(4)

    # Create Mini Test
    mini_test = orig_test.head(2)

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Point Config to mini metadata
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"  Train subset size: {len(mini_train)}")
    print(f"  Val subset size: {len(mini_val)}")
    print(f"  Test subset size: {len(mini_test)}")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architectures (Unit Tests)
    # -------------------------------------------------------------------------
    print("[3/6] Verifying Model Architectures...")
    device = Config.DEVICE

    # --- Stage 1: U-Net ---
    print("  Testing AnatomicalLocalizer (Stage 1)...")
    model_s1 = AnatomicalLocalizer(pretrained=False).to(device)
    # Input: (Batch, 1, 256, 256)
    dummy_input_s1 = torch.randn(2, 1, 256, 256).to(device)
    mask_logits, presence_probs = model_s1(dummy_input_s1)

    assert mask_logits.shape == (
        2,
        8,
        256,
        256,
    ), f"Stage 1 Mask Shape mismatch: {mask_logits.shape}"
    assert presence_probs.shape == (
        2,
        7,
    ), f"Stage 1 Presence Shape mismatch: {presence_probs.shape}"

    # --- Stage 2: Encoder ---
    print("  Testing DualBranchEncoder (Stage 2)...")
    model_s2 = DualBranchEncoder(pretrained=False).to(device)
    # Inputs: Local (Batch, 4, 224, 224), Global (Batch, 3, 224, 224)
    dummy_local = torch.randn(2, 4, 224, 224).to(device)
    dummy_global = torch.randn(2, 3, 224, 224).to(device)
    emb, logits = model_s2(dummy_local, dummy_global)

    assert emb.shape == (2, 512), f"Stage 2 Embedding Shape mismatch: {emb.shape}"
    assert logits.shape == (2, 1), f"Stage 2 Logits Shape mismatch: {logits.shape}"

    # --- Stage 3: Aggregator ---
    print("  Testing HierarchicalAggregator (Stage 3)...")
    model_s3 = HierarchicalAggregator().to(device)
    # Input: (Batch, Seq_Len, Feature_Dim + 7) = (2, 10, 519)
    dummy_seq = torch.randn(2, 10, 519).to(device)
    out_s3 = model_s3(dummy_seq)

    assert out_s3.shape == (2, 8), f"Stage 3 Output Shape mismatch: {out_s3.shape}"

    # --- Loss Function ---
    print("  Testing WeightedLogLoss...")
    criterion = WeightedLogLoss()
    loss = criterion(torch.randn(2, 8), torch.zeros(2, 8))
    assert not torch.isnan(loss), "Loss returned NaN"

    # -------------------------------------------------------------------------
    # 4. Run Training Pipelines (Integration Tests)
    # -------------------------------------------------------------------------
    print("[4/6] Running Training Pipelines (Integration Tests)...")

    # --- Stage 1 Training ---
    print("  Running Stage 1 (Segmentation) pipeline...")
    # This will process segmentation data from scratch using the mini dataset
    s1_trainer = Stage1Trainer()
    s1_trainer.train()

    s1_ckpt = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
    assert os.path.exists(s1_ckpt), "Stage 1 checkpoint not created."

    # --- Stage 2 Training ---
    print("  Running Stage 2 (Encoder) pipeline...")
    # This will process classification data (pos/neg sampling)
    s2_trainer = Stage2Trainer()
    s2_trainer.train()

    s2_ckpt = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
    assert os.path.exists(s2_ckpt), "Stage 2 checkpoint not created."

    # --- Stage 3 Training ---
    print("  Running Stage 3 (Aggregator) pipeline...")
    # This includes feature extraction which uses the trained Stage 1 & 2 models
    s3_trainer = Stage3Trainer()
    s3_trainer.train()

    s3_ckpt = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    assert os.path.exists(s3_ckpt), "Stage 3 checkpoint not created."

    # -------------------------------------------------------------------------
    # 5. Run Submission Generation
    # -------------------------------------------------------------------------
    print("[5/6] Generating Submission...")

    sub_gen = SubmissionGenerator()
    sub_gen.generate()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission generated with {len(df_sub)} rows.")
    assert len(df_sub) > 0, "Submission file is empty."
    assert "row_id" in df_sub.columns, "row_id column missing."
    assert "fractured" in df_sub.columns, "fractured column missing."

    # -------------------------------------------------------------------------
    # 6. Completion
    # -------------------------------------------------------------------------
    print("[6/6] Demo completed successfully!")
    print(f"All outputs stored in: {Config.WORKING_DIR}")


if __name__ == "__main__":
    run_demo()
