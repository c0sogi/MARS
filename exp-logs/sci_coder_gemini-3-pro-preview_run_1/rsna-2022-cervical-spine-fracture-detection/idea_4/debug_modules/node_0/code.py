import os
import pandas as pd
import numpy as np
import torch
import glob
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import weighted_log_loss
from library.models import SegmentationUNet, MaskConditionedCNN, AttentionalRNN
from library.datasets import (
    SegmentationDataset,
    CroppedSliceDataset,
    FeatureSequenceDataset,
)
from library.segmentation_engine import train_segmenter, generate_dataset_masks
from library.encoder_engine import train_encoder, extract_patient_features
from library.sequence_engine import train_aggregator, predict_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("===============================================================")
    print("   Cervical Spine Fracture Detection - End-to-End Demo")
    print("===============================================================")

    # 1. Setup and Configuration Overrides
    # -----------------------------------------------------------------
    print("\n[1] Setting up environment and configuration...")

    # Set seeds for reproducibility
    Config.seed_everything(Config.SEED)

    # Override Config for speed (Demo Mode)
    Config.SEG_EPOCHS = 1
    Config.CLS_EPOCHS = 1
    Config.SEQ_EPOCHS = 1
    Config.SEG_BATCH_SIZE = 2
    Config.CLS_BATCH_SIZE = 4
    Config.SEQ_BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Preparation (Subsetting)
    # -----------------------------------------------------------------
    print("\n[2] Loading and subsetting metadata...")

    # Load full metadata
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create subsets for the demo
    # For segmentation, we need samples with 'has_segmentation' == True
    seg_train_subset = full_train_df[full_train_df["has_segmentation"] == True].head(4)
    seg_val_subset = full_train_df[full_train_df["has_segmentation"] == True].iloc[4:6]

    # For classification/sequence, we can use any samples.
    # We'll use a mix of positive and negative cases if possible.
    # Taking top 10 samples for general training
    train_subset = full_train_df.head(10)
    val_subset = full_val_df.head(4)

    # Use validation subset as a proxy for test set for inference demo
    test_subset = full_val_df.head(4).copy()

    print(f"Segmentation Train Subset: {len(seg_train_subset)} samples")
    print(f"General Train Subset: {len(train_subset)} samples")
    print(f"Test Subset: {len(test_subset)} samples")

    # 3. Stage 1: Segmentation (U-Net)
    # -----------------------------------------------------------------
    print("\n[3] Stage 1: Segmentation (Spine Localization)...")

    # A. Model Verification
    print("   -> Verifying Segmentation Model logic...")
    model_seg = SegmentationUNet(n_channels=3, n_classes=1).to(Config.DEVICE)
    dummy_input = torch.randn(2, 3, 512, 512).to(Config.DEVICE)
    with torch.no_grad():
        dummy_out = model_seg(dummy_input)
    assert dummy_out.shape == (
        2,
        1,
        512,
        512,
    ), f"Seg output shape mismatch: {dummy_out.shape}"
    print("      Segmentation model forward pass successful.")

    # B. Training
    print("   -> Training Segmentation Model (Demo)...")
    if len(seg_train_subset) > 0:
        train_segmenter(seg_train_subset, seg_val_subset)
        assert os.path.exists(
            os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")
        ), "Segmentation checkpoint not found after training."
    else:
        print("      Skipping training (no segmentation data in subset).")

    # C. Mask Generation (Inference)
    print("   -> Generating Masks for downstream stages...")
    # We generate masks for the 'train_subset' and 'val_subset' so Stage 2 can use them
    # Combine unique UIDs
    all_uids_df = pd.concat([train_subset, val_subset, test_subset]).drop_duplicates(
        subset="StudyInstanceUID"
    )
    mask_dir = generate_dataset_masks(all_uids_df, load_cached_data=False)

    # Verify masks were created
    sample_uid = all_uids_df.iloc[0]["StudyInstanceUID"]
    sample_mask_path = os.path.join(mask_dir, sample_uid)
    if os.path.exists(sample_mask_path) and len(os.listdir(sample_mask_path)) > 0:
        print(f"      Verified masks generated for {sample_uid}")
    else:
        # It's possible the sample had no valid DICOMs or failed, but we expect success here.
        # We won't crash, but we note it.
        print(f"      Warning: No masks found for {sample_uid}. Check input data.")

    # 4. Stage 2: Slice Encoder (2.5D CNN)
    # -----------------------------------------------------------------
    print("\n[4] Stage 2: Slice Encoder (Fracture Classification)...")

    # A. Dataset Verification
    print("   -> Verifying CroppedSliceDataset logic...")
    # Initialize dataset with the generated masks
    ds_encoder = CroppedSliceDataset(train_subset, mode="train", mask_dir=mask_dir)
    if len(ds_encoder) > 0:
        img_tensor, label_tensor = ds_encoder[0]
        # Expected shape: (C, H, W). C = 3 (RGB) + 1 (Mask) = 4 if Config.USE_MASK_INPUT is True
        expected_channels = 4 if Config.USE_MASK_INPUT else 3
        assert img_tensor.shape == (
            expected_channels,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), f"Encoder input shape mismatch: {img_tensor.shape}"
        assert isinstance(label_tensor.item(), float), "Label should be a float scalar"
        print(f"      Dataset item shape: {img_tensor.shape}, Label: {label_tensor}")
    else:
        print(
            "      Warning: Encoder dataset is empty (possibly no DICOMs found in subset paths)."
        )

    # B. Model Verification
    print("   -> Verifying Encoder Model logic...")
    model_enc = MaskConditionedCNN(pretrained=False).to(Config.DEVICE)
    dummy_enc_in = torch.randn(2, 4, 256, 256).to(Config.DEVICE)  # 4 channels
    with torch.no_grad():
        dummy_logits = model_enc(dummy_enc_in)  # Classification head
        dummy_feats = model_enc.forward_features(dummy_enc_in)  # Feature head
    assert dummy_logits.shape == (
        2,
        1,
    ), f"Encoder logits shape mismatch: {dummy_logits.shape}"
    assert dummy_feats.shape == (
        2,
        Config.ENCODER_HIDDEN_DIM,
    ), f"Encoder features shape mismatch: {dummy_feats.shape}"
    print("      Encoder model forward pass successful.")

    # C. Training
    print("   -> Training Encoder Model (Demo)...")
    train_encoder(train_subset, val_subset, mask_dir)
    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "slice_encoder.pth")
    ), "Encoder checkpoint not found after training."

    # D. Feature Extraction
    print("   -> Extracting Features for Sequence Stage...")
    feature_dir = extract_patient_features(
        all_uids_df, mask_dir, load_cached_data=False
    )

    # Verify features
    if len(all_uids_df) > 0:
        sample_feat_path = os.path.join(
            feature_dir, f"{all_uids_df.iloc[0]['StudyInstanceUID']}.npy"
        )
        assert os.path.exists(sample_feat_path), "Feature file not created."
        feat_arr = np.load(sample_feat_path)
        print(f"      Verified feature extraction. Shape: {feat_arr.shape}")

    # 5. Stage 3: Sequence Aggregator (RNN)
    # -----------------------------------------------------------------
    print("\n[5] Stage 3: Sequence Aggregator (Patient Level)...")

    # A. Dataset Verification
    print("   -> Verifying FeatureSequenceDataset logic...")
    ds_seq = FeatureSequenceDataset(train_subset, feature_dir)
    if len(ds_seq) > 0:
        feats, mask, labels = ds_seq[0]
        # feats: (MAX_SEQ_LEN, HIDDEN_DIM)
        assert feats.shape == (
            Config.MAX_SEQ_LEN,
            Config.ENCODER_HIDDEN_DIM,
        ), f"Sequence feature shape mismatch: {feats.shape}"
        assert mask.shape == (
            Config.MAX_SEQ_LEN,
        ), f"Sequence mask shape mismatch: {mask.shape}"
        assert labels.shape == (8,), f"Sequence label shape mismatch: {labels.shape}"
        print("      Sequence dataset verification successful.")

    # B. Model Verification
    print("   -> Verifying RNN Aggregator logic...")
    model_rnn = AttentionalRNN(
        input_dim=Config.ENCODER_HIDDEN_DIM, hidden_dim=Config.SEQ_HIDDEN_DIM
    ).to(Config.DEVICE)
    dummy_seq = torch.randn(2, Config.MAX_SEQ_LEN, Config.ENCODER_HIDDEN_DIM).to(
        Config.DEVICE
    )
    dummy_mask = torch.ones(2, Config.MAX_SEQ_LEN).to(Config.DEVICE)
    with torch.no_grad():
        rnn_out = model_rnn(dummy_seq, dummy_mask)
    assert rnn_out.shape == (2, 8), f"RNN output shape mismatch: {rnn_out.shape}"
    print("      RNN model forward pass successful.")

    # C. Training
    print("   -> Training Aggregator Model (Demo)...")
    train_aggregator(train_subset, val_subset, feature_dir)
    assert os.path.exists(
        os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    ), "Aggregator checkpoint not found after training."

    # 6. Inference Pipeline
    # -----------------------------------------------------------------
    print("\n[6] Running Final Inference Pipeline...")

    # We use the test_subset (which is just a slice of validation data here)
    submission_df = predict_pipeline(test_subset, feature_dir)

    # Verification
    assert os.path.exists(
        Config.SAMPLE_SUBMISSION_PATH
    ), "Sample submission file missing (input check)."
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(output_path), "Submission file was not created."

    print("\n[7] Submission Preview:")
    print(submission_df.head(10))

    # Check metric calculation on dummy data
    print("\n[8] Metric Verification (Weighted Log Loss)...")
    # Create dummy true labels matching submission shape
    y_true = np.random.randint(0, 2, (5, 8)).astype(np.float32)
    y_pred = np.random.rand(5, 8).astype(np.float32)
    loss = weighted_log_loss(y_true, y_pred)
    print(f"      Calculated dummy loss: {loss:.4f}")
    assert isinstance(loss, float), "Loss should be a float."

    print("\n===============================================================")
    print("   Demo Completed Successfully.")
    print("===============================================================")


if __name__ == "__main__":
    run_demo()
