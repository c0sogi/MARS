import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_from_cache
from library.models import UNetLocalizer, DetailEncoder, HierarchicalRNN
from library.datasets import (
    SegmentationDataset,
    CropClassificationDataset,
    SequenceDataset,
)
from library.engine import (
    train_segmentor,
    generate_stage1_results,
    train_encoder,
    extract_features,
    train_aggregator,
    generate_submission,
)
from library.feature_extractor import load_stage1_model, load_stage2_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting End-to-End Pipeline Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set Hyperparameters for speed
    Config.STAGE1_EPOCHS = 1
    Config.STAGE1_BATCH_SIZE = 2
    Config.STAGE2_EPOCHS = 1
    Config.STAGE2_BATCH_SIZE = 2
    Config.STAGE3_EPOCHS = 1
    Config.STAGE3_BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Subset Selection)
    # -------------------------------------------------------------------------
    print("\n--- Data Preparation ---")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select a small subset of studies that have bounding boxes for Stage 1
    # We check 'has_bounding_box' column because we replaced NIFTI segmentations with BBoxes
    seg_studies = train_meta[train_meta["has_bounding_box"] == True]
    if len(seg_studies) == 0:
        print(
            "No studies with bounding boxes found in train metadata. Using first 2 rows."
        )
        subset_train = train_meta.head(2)
    else:
        subset_train = seg_studies.head(2)

    # Ensure validation subset also has bounding boxes if possible
    val_bbox_studies = val_meta[val_meta["has_bounding_box"] == True]
    if len(val_bbox_studies) > 0:
        subset_val = val_bbox_studies.head(2)
    else:
        subset_val = val_meta.head(2)
    subset_test = test_meta.head(2)

    print(f"Selected Train Subset: {len(subset_train)} studies")
    print(f"Selected Val Subset: {len(subset_val)} studies")

    # -------------------------------------------------------------------------
    # 3. Stage 1: Segmentation (UNetLocalizer)
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Segmentation ---")

    # Initialize Model
    model_s1 = UNetLocalizer(
        in_channels=Config.STAGE1_IN_CHANNELS, num_classes=Config.STAGE1_NUM_CLASSES
    ).to(device)

    optimizer_s1 = optim.Adam(model_s1.parameters(), lr=Config.STAGE1_LR)

    # Initialize Datasets
    # Note: SegmentationDataset prepares cache internally.
    # We force load_cached_data=False to demonstrate processing logic.
    train_ds_s1 = SegmentationDataset(
        subset_train, mode="train", load_cached_data=False
    )
    val_ds_s1 = SegmentationDataset(subset_val, mode="val", load_cached_data=False)

    if len(train_ds_s1) > 0:
        train_loader_s1 = DataLoader(
            train_ds_s1,
            batch_size=Config.STAGE1_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        val_loader_s1 = DataLoader(
            val_ds_s1, batch_size=Config.STAGE1_BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Train
        train_segmentor(
            train_loader_s1,
            val_loader_s1,
            model_s1,
            optimizer_s1,
            device,
            epochs=Config.STAGE1_EPOCHS,
        )
    else:
        print(
            "Skipping Stage 1 training (no segmentation data found/loaded in subset)."
        )

    # Inference (Localization Metadata Generation)
    # We run this on the training subset to generate ROI/Maps for Stage 2 training demo
    stage1_results_df = generate_stage1_results(
        subset_train, model_s1, device, load_cached_data=False
    )

    # Verify Stage 1 Output
    assert not stage1_results_df.empty, "Stage 1 results DataFrame is empty"
    assert "roi_y" in stage1_results_df.columns, "ROI Y coordinate missing"
    assert "anatomical_map" in stage1_results_df.columns, "Anatomical map missing"
    print("Stage 1 verification passed.")

    # -------------------------------------------------------------------------
    # 4. Stage 2: Feature Encoder (DetailEncoder)
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Feature Extraction ---")

    # Initialize Model
    model_s2 = DetailEncoder(in_channels=Config.STAGE2_IN_CHANNELS).to(device)
    optimizer_s2 = optim.Adam(model_s2.parameters(), lr=Config.STAGE2_LR)

    # Prepare Slice-Level Data for Classification Demo
    # We expand the study-level metadata to slice-level using Stage 1 results
    # For demo purposes, we assign the patient-level label to these slices (simplification)

    def prepare_stage2_df(meta_df, s1_results):
        records = []
        for _, row in meta_df.iterrows():
            uid = row["StudyInstanceUID"]
            # Get slices from stage 1 results
            s1_subset = s1_results[s1_results["StudyInstanceUID"] == uid]
            if s1_subset.empty:
                continue

            # Take a few slices per study for demo
            for _, s_row in s1_subset.head(5).iterrows():
                records.append(
                    {
                        "StudyInstanceUID": uid,
                        "slice_index": s_row["slice_index"],
                        "image_path": row["image_path"],
                        "label": row["patient_overall"],  # Simplified label assignment
                        "mask_file": s_row["mask_file"],
                    }
                )
        return pd.DataFrame(records)

    train_df_s2 = prepare_stage2_df(subset_train, stage1_results_df)

    # Create ROI Map for cropping
    roi_map = {}
    for _, row in stage1_results_df.iterrows():
        uid = row["StudyInstanceUID"]
        sl_idx = row["slice_index"]
        if uid not in roi_map:
            roi_map[uid] = {}
        roi_map[uid][sl_idx] = [int(row["roi_y"]), int(row["roi_x"])]

    if not train_df_s2.empty:
        train_ds_s2 = CropClassificationDataset(
            train_df_s2, mode="train", roi_map=roi_map
        )
        # Split same dataset for val demo
        val_ds_s2 = CropClassificationDataset(
            train_df_s2.iloc[:2], mode="val", roi_map=roi_map
        )

        train_loader_s2 = DataLoader(
            train_ds_s2,
            batch_size=Config.STAGE2_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        val_loader_s2 = DataLoader(
            val_ds_s2, batch_size=Config.STAGE2_BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Train
        train_encoder(
            train_loader_s2,
            val_loader_s2,
            model_s2,
            optimizer_s2,
            device,
            epochs=Config.STAGE2_EPOCHS,
        )
    else:
        print("Skipping Stage 2 training (no slice data generated).")

    # Feature Extraction
    # Extract features for the subset_train to use in Stage 3
    feature_dir = extract_features(
        subset_train, stage1_results_df, model_s2, device, load_cached_data=False
    )

    # Verify Features
    sample_feat_path = os.path.join(
        feature_dir, f"{subset_train.iloc[0]['StudyInstanceUID']}.npy"
    )
    assert os.path.exists(sample_feat_path), "Feature file not generated"
    data = np.load(sample_feat_path, allow_pickle=True).item()
    assert (
        data["features"].shape[1] == Config.STAGE2_FEATURE_DIM
    ), "Incorrect feature dimension"
    print("Stage 2 verification passed.")

    # -------------------------------------------------------------------------
    # 5. Stage 3: Aggregator (HierarchicalRNN)
    # -------------------------------------------------------------------------
    print("\n--- Stage 3: Aggregation ---")

    # Initialize Model
    model_s3 = HierarchicalRNN(
        input_dim=Config.STAGE3_INPUT_DIM, hidden_dim=Config.STAGE3_HIDDEN_DIM
    ).to(device)

    optimizer_s3 = optim.Adam(model_s3.parameters(), lr=Config.STAGE3_LR)

    # Initialize Dataset
    # We use the features extracted from subset_train
    train_ds_s3 = SequenceDataset(subset_train, feature_dir, max_len=100)
    val_ds_s3 = SequenceDataset(
        subset_train, feature_dir, max_len=100
    )  # Reuse for demo

    train_loader_s3 = DataLoader(
        train_ds_s3, batch_size=Config.STAGE3_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader_s3 = DataLoader(
        val_ds_s3, batch_size=Config.STAGE3_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Train
    train_aggregator(
        train_loader_s3,
        val_loader_s3,
        model_s3,
        optimizer_s3,
        device,
        epochs=Config.STAGE3_EPOCHS,
    )

    # Submission / Inference
    # We need to extract features for test set first.
    # For demo speed, we will just use the train subset as "test" to prove the submission generation works.
    print("Generating submission using training subset as proxy for test...")
    test_loader_s3 = DataLoader(
        val_ds_s3, batch_size=Config.STAGE3_BATCH_SIZE, shuffle=False, num_workers=0
    )

    generate_submission(model_s3, test_loader_s3, subset_train, device)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "row_id" in sub_df.columns and "fractured" in sub_df.columns
    ), "Submission columns incorrect"
    assert (
        len(sub_df) == len(subset_train) * 8
    ), "Incorrect number of rows in submission"
    print("Stage 3 verification passed.")

    print("\nEnd-to-End Pipeline Demonstration Completed Successfully.")


if __name__ == "__main__":
    run_demo()
