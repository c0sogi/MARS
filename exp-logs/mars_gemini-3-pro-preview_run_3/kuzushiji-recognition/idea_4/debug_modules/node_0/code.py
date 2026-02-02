import os
import sys
import pandas as pd
import torch
import numpy as np
import cv2
import shutil

# Library imports
from library.config import Config
from library.utils import set_seed
from library.dataset import (
    get_class_map,
    prepare_classifier_data,
    PatchDetectorDataset,
    CharacterCropDataset,
    get_transforms,
)
from library.models import CenterNetDetector, ResNetClassifier
from library.losses import ModifiedFocalLoss, RegL1Loss
from library.trainer import DetectorTrainer, ClassifierTrainer
from library.inference import TiledDetector, InferencePipeline


def main():
    print("=== Starting Kuzushiji Recognition Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed/Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Enable debug mode to use small data subsets (20 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # Minimize training duration
    Config.DETECTOR_EPOCHS = 1
    Config.CLASSIFIER_EPOCHS = 1

    # Adjust batch sizes for the demo environment
    Config.DETECTOR_BATCH_SIZE = 2
    Config.CLASSIFIER_BATCH_SIZE = 4

    # Disable multiprocessing to avoid overhead in this short script
    Config.NUM_WORKERS = 0

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print("Configuration set to DEBUG mode (small subset, 1 epoch).")

    # ---------------------------------------------------------
    # 2. Dataset Utilities & Preparation
    # ---------------------------------------------------------
    print("\n[2] Testing Dataset Utilities...")

    # Load full metadata (will be sliced later or used for map generation)
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    print(f"Loaded training metadata with {len(train_df)} rows.")

    # Test Class Map Generation
    # We use a small subset to generate a map quickly for the demo
    subset_df = train_df.head(50)
    class_map = get_class_map(subset_df, save_path=Config.CLASS_MAP_PATH)
    print(f"Generated class map with {len(class_map)} classes from subset.")
    assert isinstance(class_map, dict), "Class map should be a dictionary"

    # Test Classifier Data Preparation
    # This parses the label strings into individual character samples
    samples = prepare_classifier_data(
        subset_df,
        class_map,
        split_name="demo_train",
        cache_dir=Config.CACHE_DIR,
        load_cached_data=False,
    )
    print(f"Parsed {len(samples)} character samples from subset.")
    if len(samples) > 0:
        s = samples[0]
        assert "image_path" in s and "class_id" in s and "bbox" in s
        assert len(s["bbox"]) == 4

    # ---------------------------------------------------------
    # 3. Dataset Classes
    # ---------------------------------------------------------
    print("\n[3] Testing Dataset Classes...")

    # PatchDetectorDataset
    det_ds = PatchDetectorDataset(
        subset_df,
        mode="train_detector",
        transform=get_transforms("train_detector", Config.DETECTOR_INPUT_SIZE),
    )
    det_img, det_target = det_ds[0]

    # Verify Detector Shapes
    # Image: (3, H, W)
    assert det_img.shape == (3, Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE)

    # Heatmap: (1, H/4, W/4)
    out_size = Config.DETECTOR_INPUT_SIZE // Config.DETECTOR_STRIDE
    assert det_target["hm"].shape == (1, out_size, out_size)
    assert det_target["wh"].shape == (2, out_size, out_size)
    assert det_target["reg"].shape == (2, out_size, out_size)
    print("PatchDetectorDataset output shapes verified.")

    # CharacterCropDataset
    cls_ds = CharacterCropDataset(
        subset_df,
        class_map,
        split_name="demo_cls",
        mode="train_classifier",
        transform=get_transforms("train_classifier", Config.CLASSIFIER_INPUT_SIZE),
        cache_images=False,  # Disable RAM cache for demo to save memory
    )
    if len(cls_ds) > 0:
        cls_img, cls_label = cls_ds[0]
        # Image: (3, 128, 128)
        assert cls_img.shape == (
            3,
            Config.CLASSIFIER_INPUT_SIZE,
            Config.CLASSIFIER_INPUT_SIZE,
        )
        assert isinstance(cls_label, torch.Tensor)
        print("CharacterCropDataset output shapes verified.")
    else:
        print("Warning: No samples found in subset for classifier dataset.")

    # ---------------------------------------------------------
    # 4. Models & Losses
    # ---------------------------------------------------------
    print("\n[4] Testing Models and Losses...")
    device = Config.DEVICE

    # -- Detector Model --
    det_model = CenterNetDetector(pretrained=False).to(device)
    # Create dummy batch
    dummy_det_in = torch.randn(
        2, 3, Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE
    ).to(device)

    # Forward pass
    hm, wh, reg = det_model(dummy_det_in)

    assert hm.shape == (2, 1, out_size, out_size)
    assert wh.shape == (2, 2, out_size, out_size)
    assert reg.shape == (2, 2, out_size, out_size)
    print("CenterNetDetector forward pass successful.")

    # -- Detector Losses --
    hm_loss_fn = ModifiedFocalLoss()
    reg_loss_fn = RegL1Loss()

    # Dummy targets
    t_hm = torch.zeros_like(hm)
    t_wh = torch.zeros_like(wh)
    t_mask = torch.ones((2, 1, out_size, out_size)).to(device)

    l_hm = hm_loss_fn(hm, t_hm)
    l_wh = reg_loss_fn(wh, t_wh, t_mask)

    assert not torch.isnan(l_hm)
    assert not torch.isnan(l_wh)
    print("Detector loss calculation successful.")

    # -- Classifier Model --
    num_classes = len(class_map) if len(class_map) > 0 else 5
    cls_model = ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)

    dummy_cls_in = torch.randn(
        2, 3, Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE
    ).to(device)
    logits = cls_model(dummy_cls_in)

    assert logits.shape == (2, num_classes)
    print("ResNetClassifier forward pass successful.")

    # ---------------------------------------------------------
    # 5. Training Loops (Trainers)
    # ---------------------------------------------------------
    print("\n[5] Testing Trainers (Mock Run)...")

    # Initialize Detector Trainer
    # Note: Config.DEBUG=True forces the trainer to use a small subset of data (20 samples)
    det_trainer = DetectorTrainer()
    print("DetectorTrainer initialized.")

    # Run 1 epoch
    det_metrics = det_trainer.train_epoch(epoch=1)
    print(f"Detector Train Metrics: {det_metrics}")

    # Validate
    det_val_loss = det_trainer.validate()
    print(f"Detector Val Loss: {det_val_loss}")

    # Save checkpoint manually for inference step (simulating `fit` saving best model)
    torch.save(det_model.state_dict(), Config.DETECTOR_CHECKPOINT)
    print("Saved dummy detector checkpoint.")

    # Initialize Classifier Trainer
    cls_trainer = ClassifierTrainer()
    print("ClassifierTrainer initialized.")

    # Run 1 epoch
    cls_metrics = cls_trainer.train_epoch(epoch=1)
    print(f"Classifier Train Metrics: {cls_metrics}")

    # Save checkpoint
    torch.save(cls_model.state_dict(), Config.CLASSIFIER_CHECKPOINT)
    print("Saved dummy classifier checkpoint.")

    # ---------------------------------------------------------
    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[6] Testing Inference Pipeline...")

    # Create a mini test metadata file with 2 images
    test_df_full = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test_df = test_df_full.head(2).copy()
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")
    mini_test_df.to_csv(mini_test_path, index=False)

    # Initialize Pipeline
    # This loads the checkpoints we just saved
    pipeline = InferencePipeline()

    # Run inference
    output_sub_path = os.path.join(Config.WORKING_DIR, "mini_submission.csv")
    pipeline.run(test_metadata_path=mini_test_path, output_path=output_sub_path)

    # Verify Output
    if os.path.exists(output_sub_path):
        sub_df = pd.read_csv(output_sub_path)
        print(f"Submission generated with {len(sub_df)} rows.")
        assert len(sub_df) == 2
        assert "image_id" in sub_df.columns
        assert "labels" in sub_df.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
