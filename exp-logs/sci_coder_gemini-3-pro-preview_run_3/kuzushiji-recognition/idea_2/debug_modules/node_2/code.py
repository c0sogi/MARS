import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import (
    gaussian_radius,
    gaussian2D,
    draw_umich_gaussian,
    f1_score_calc,
    decode_detections,
    collate_fn_detector,
    collate_fn_classifier,
)
from library.models import CenterNetDetector, CharacterClassifier
from library.dataset import (
    KuzushijiDetectorDataset,
    KuzushijiCropDataset,
    get_class_map,
    process_detector_metadata,
)
from library.preprocess import prepare_classifier_data
from library.engine import (
    train_one_epoch_detector,
    train_one_epoch_classifier,
    ModifiedFocalLoss,
    RegL1Loss,
)
from library.inference import InferencePipeline


def test_utils():
    print("\n=== Testing Utils ===")

    # 1. Gaussian Radius
    size = (100, 100)
    radius = gaussian_radius(size, min_overlap=0.7)
    assert radius > 0, "Gaussian radius should be positive"
    print(f"Gaussian radius for 100x100: {radius:.2f}")

    # 2. Gaussian 2D
    g2d = gaussian2D((5, 5), sigma=1)
    assert g2d.shape == (5, 5)
    assert np.isclose(g2d.max(), 1.0), "Gaussian max should be 1.0"
    print("Gaussian 2D generation successful")

    # 3. F1 Score
    # Perfect match scenario
    preds = [{"label": "A", "x": 10, "y": 10, "score": 0.9}]
    # GT Box: x=5, y=5, w=10, h=10 (covers 5,5 to 15,15). Point 10,10 is inside.
    targets = [{"label": "A", "x": 5, "y": 5, "w": 10, "h": 10}]
    scores = f1_score_calc([preds], [targets])
    assert scores["f1"] == 1.0, f"Expected F1 1.0, got {scores['f1']}"

    # No match scenario (wrong label)
    preds_wrong = [{"label": "B", "x": 10, "y": 10, "score": 0.9}]
    scores_wrong = f1_score_calc([preds_wrong], [targets])
    assert scores_wrong["f1"] == 0.0, f"Expected F1 0.0, got {scores_wrong['f1']}"

    print("F1 Score calculation verified")


def setup_mini_data():
    print("\n=== Setting up Mini Dataset ===")

    # Create a temporary working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # Update Config to use this directory for outputs and cache
    Config.WORKING_DIR = demo_dir
    Config.CACHE_CLASS_MAP = os.path.join(demo_dir, "class_map.npy")
    Config.CACHE_DETECTOR_TRAIN = os.path.join(demo_dir, "detector_train.npy")
    Config.CACHE_DETECTOR_VAL = os.path.join(demo_dir, "detector_val.npy")
    Config.CACHE_CLASSIFIER_TRAIN = os.path.join(demo_dir, "classifier_train.npy")
    Config.CACHE_CLASSIFIER_VAL = os.path.join(demo_dir, "classifier_val.npy")
    Config.DETECTOR_MODEL_PATH = os.path.join(demo_dir, "detector_best.pth")
    Config.CLASSIFIER_MODEL_PATH = os.path.join(demo_dir, "classifier_best.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Create mini metadata files by sampling the real ones.
    # We use the real paths so image loading works, but only take a few rows.

    # Train Metadata
    real_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    mini_train = real_train_meta.head(10).copy()  # Take 10 samples
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_train.to_csv(mini_train_path, index=False)
    Config.TRAIN_METADATA_PATH = mini_train_path

    # Val Metadata
    real_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    mini_val = real_val_meta.head(5).copy()
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_val.to_csv(mini_val_path, index=False)
    Config.VAL_METADATA_PATH = mini_val_path

    # Test Metadata
    real_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test = real_test_meta.head(5).copy()
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    mini_test.to_csv(mini_test_path, index=False)
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"Mini datasets created in {demo_dir}")


def test_preprocessing():
    print("\n=== Testing Preprocessing ===")

    # Test Class Map generation
    char_to_idx, idx_to_char = get_class_map(load_cached=False)
    assert len(char_to_idx) > 0
    print(f"Class map generated with {len(char_to_idx)} classes")

    # Test Classifier Data Preparation (Crops)
    # This reads the mini_train.csv we set up and extracts crops from real images
    data = prepare_classifier_data(split="train", load_cached=False)
    assert len(data) > 0, "No crops generated from mini dataset"
    assert os.path.exists(data[0]["image_path"]), "Crop image file does not exist"
    print(f"Generated {len(data)} crops for classifier training")


def test_datasets():
    print("\n=== Testing Datasets ===")

    # 1. Detector Dataset
    # debug=False because we already manually created a mini dataset via metadata
    ds_det = KuzushijiDetectorDataset(split="train", debug=False, load_cached=False)
    item_det = ds_det[0]

    assert item_det is not None
    assert "img" in item_det
    assert "heatmap" in item_det
    assert item_det["img"].shape == (
        3,
        Config.DETECTOR_IMG_SIZE,
        Config.DETECTOR_IMG_SIZE,
    )

    # Heatmap should be 1/4 size (stride 4)
    out_size = Config.DETECTOR_IMG_SIZE // Config.DETECTOR_OUTPUT_STRIDE
    assert item_det["heatmap"].shape == (1, out_size, out_size)
    assert item_det["heatmap"].max() <= 1.0
    print("Detector Dataset loaded successfully")

    # 2. Classifier Dataset
    ds_cls = KuzushijiCropDataset(split="train", debug=False, load_cached=True)
    if len(ds_cls) > 0:
        item_cls = ds_cls[0]
        assert item_cls is not None
        assert "img" in item_cls
        assert "label_idx" in item_cls
        assert item_cls["img"].shape == (
            3,
            Config.CLASSIFIER_IMG_SIZE,
            Config.CLASSIFIER_IMG_SIZE,
        )
        print("Classifier Dataset loaded successfully")
    else:
        print(
            "Warning: Classifier dataset empty (might happen if mini dataset has no labels)"
        )


def test_models_and_training():
    print("\n=== Testing Models and Training Engine ===")

    device = "cpu"  # Use CPU for quick demo to avoid CUDA initialization overhead
    Config.DEVICE = device

    # --- Detector ---
    print("Initializing Detector...")
    model_det = CenterNetDetector(pretrained=False).to(device)

    # Create dummy batch to verify forward pass
    B = 2
    dummy_input = torch.randn(
        B, 3, Config.DETECTOR_IMG_SIZE, Config.DETECTOR_IMG_SIZE
    ).to(device)

    # Forward
    out = model_det(dummy_input)
    assert "heatmap" in out
    assert out["heatmap"].shape == (B, 1, 256, 256)  # 1024 / 4

    # Training Step
    ds_det = KuzushijiDetectorDataset(split="train", debug=False, load_cached=True)
    loader_det = DataLoader(ds_det, batch_size=2, collate_fn=collate_fn_detector)

    optimizer_det = torch.optim.Adam(model_det.parameters(), lr=1e-4)
    criterion_hm = ModifiedFocalLoss().to(device)
    criterion_reg = RegL1Loss().to(device)

    print("Running Detector Training Step...")
    # Run just one epoch (which is very short due to mini dataset)
    metrics = train_one_epoch_detector(
        model_det, loader_det, optimizer_det, criterion_hm, criterion_reg, device
    )
    assert "loss" in metrics
    print(f"Detector Train Loss: {metrics['loss']:.4f}")

    # Save dummy model for inference test
    torch.save(model_det.state_dict(), Config.DETECTOR_MODEL_PATH)

    # --- Classifier ---
    print("Initializing Classifier...")
    model_cls = CharacterClassifier(
        num_classes=Config.NUM_CLASSES, pretrained=False
    ).to(device)

    # Dummy input
    dummy_crop = torch.randn(
        B, 3, Config.CLASSIFIER_IMG_SIZE, Config.CLASSIFIER_IMG_SIZE
    ).to(device)
    logits = model_cls(dummy_crop)
    assert logits.shape == (B, Config.NUM_CLASSES)

    # Training Step
    ds_cls = KuzushijiCropDataset(split="train", debug=False, load_cached=True)
    if len(ds_cls) > 1:
        loader_cls = DataLoader(ds_cls, batch_size=2, collate_fn=collate_fn_classifier)
        optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=1e-3)
        criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        print("Running Classifier Training Step...")
        loss, acc = train_one_epoch_classifier(
            model_cls, loader_cls, optimizer_cls, criterion_cls, device
        )
        print(f"Classifier Train Loss: {loss:.4f}, Acc: {acc:.4f}")

        # Save dummy model
        torch.save(model_cls.state_dict(), Config.CLASSIFIER_MODEL_PATH)
    else:
        print("Skipping Classifier training step (not enough data in mini-set)")


def test_inference_components():
    print("\n=== Testing Inference Components ===")

    pipeline = InferencePipeline(device="cpu")

    # Test Coordinate Correction
    # Scenario: Input size 1024. Original image 2048x1024 (Wide).
    # Albumentations LongestMaxSize(1024) -> scales to 1024x512.
    # PadIfNeeded(1024, 1024) -> Pads top/bottom to make it 1024x1024.
    # Scale = 0.5. Pad Top = 256.

    orig_shape = (1024, 2048)  # H, W
    input_size = 1024

    # A point at center of input image (512, 512)
    # Should correspond to center of original image (512, 1024)
    det = np.array([[512, 512, 20, 20, 0.9, 0]])

    corrected = pipeline.correct_coordinates(det, orig_shape, input_size)

    # Verification:
    # y_orig = (512 - 256) / 0.5 = 512.
    # x_orig = (512 - 0) / 0.5 = 1024.

    assert np.isclose(corrected[0, 0], 1024), f"Expected x=1024, got {corrected[0,0]}"
    assert np.isclose(corrected[0, 1], 512), f"Expected y=512, got {corrected[0,1]}"
    print("Coordinate correction verified")

    # Test Process Crops
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Detection at 50, 50 with size 10, 10
    dets = np.array([[50, 50, 10, 10, 1.0, 0]])

    crops = pipeline.process_crops(img, dets)
    assert crops is not None
    assert crops.shape == (1, 3, 64, 64)  # Should be resized to classifier input
    print("Crop processing verified")

    # Run full dummy inference
    # We need to make sure models are loaded. They were saved in test_models_and_training
    print("Running full inference pipeline on mini test set...")
    pipeline.run()

    assert os.path.exists(Config.SUBMISSION_PATH)
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_sub) == 5  # We put 5 images in mini_test
    print(f"Inference completed. Submission shape: {df_sub.shape}")


if __name__ == "__main__":
    # Set seed for reproducibility
    Config.set_seed(42)

    try:
        setup_mini_data()
        test_utils()
        test_preprocessing()
        test_datasets()
        test_models_and_training()
        test_inference_components()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
