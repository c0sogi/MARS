import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Import from library
from library.config import Config
from library.dataset import (
    KuzushijiDetectorDataset,
    KuzushijiClassifierDataset,
    get_detector_transforms,
    get_classifier_transforms,
    prepare_classifier_data,
    parse_labels,
)
from library.models import KuzushijiDetector, KuzushijiClassifier
from library.engine import fit_detector, fit_classifier, set_seed
from library.inference import TiledDetector, BatchClassifier, generate_submission
from library.utils import calc_modified_f1


def main():
    # 1. Setup and Configuration Override for Fast Baseline
    set_seed(Config.SEED)

    # Override Config for speed (Fast Baseline)
    Config.DETECTOR_EPOCHS = 5
    Config.CLASSIFIER_EPOCHS = 5
    Config.DETECTOR_BATCH_SIZE = 16  # Increase batch size for A100

    # Limits for subsampling
    MAX_DETECTOR_SAMPLES = 800
    MAX_CLASSIFIER_SAMPLES = 10000

    print(f"Running on device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading & Subsetting
    # ==========================================
    print("\n=== Preparing Data ===")

    # --- Detector Data ---
    full_train_det_ds = KuzushijiDetectorDataset(
        Config.TRAIN_METADATA, mode="train", transform=get_detector_transforms("train")
    )
    # Subsample detector data
    det_indices = list(range(len(full_train_det_ds)))
    random.shuffle(det_indices)
    train_det_indices = det_indices[:MAX_DETECTOR_SAMPLES]
    train_det_ds = Subset(full_train_det_ds, train_det_indices)

    val_det_ds = KuzushijiDetectorDataset(
        Config.VAL_METADATA, mode="val", transform=get_detector_transforms("val")
    )

    train_det_loader = DataLoader(
        train_det_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_det_loader = DataLoader(
        val_det_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Classifier Data ---
    # Load cached arrays
    train_X, train_y = prepare_classifier_data(
        Config.TRAIN_METADATA, "classifier_train", load_cached_data=True
    )
    val_X, val_y = prepare_classifier_data(
        Config.VAL_METADATA, "classifier_val", load_cached_data=True
    )

    # Subsample classifier data
    if len(train_y) > MAX_CLASSIFIER_SAMPLES:
        indices = np.random.choice(len(train_y), MAX_CLASSIFIER_SAMPLES, replace=False)
        train_X = train_X[indices]
        train_y = train_y[indices]

    # Class Balancing for Classifier
    class_counts = np.bincount(train_y)
    class_weights = np.zeros(len(class_counts))
    # Avoid division by zero
    mask = class_counts > 0
    class_weights[mask] = 1.0 / class_counts[mask]
    sample_weights = class_weights[train_y]
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_cls_ds = KuzushijiClassifierDataset(
        train_X, train_y, transform=get_classifier_transforms("train")
    )
    val_cls_ds = KuzushijiClassifierDataset(
        val_X, val_y, transform=get_classifier_transforms("val")
    )

    train_cls_loader = DataLoader(
        train_cls_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_cls_loader = DataLoader(
        val_cls_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\n=== Training Detector ===")
    detector = KuzushijiDetector(pretrained=True).to(Config.DEVICE)
    det_optimizer = optim.AdamW(detector.parameters(), lr=Config.DETECTOR_LR)

    fit_detector(
        detector,
        train_det_loader,
        val_det_loader,
        det_optimizer,
        Config.DEVICE,
        epochs=Config.DETECTOR_EPOCHS,
    )

    print("\n=== Training Classifier ===")
    # Determine num_classes from data or map
    class_map_path = Config.CLASS_MAP_PATH
    if os.path.exists(class_map_path):
        class_map = np.load(class_map_path, allow_pickle=True).item()
        num_classes = len(class_map)
    else:
        # Fallback if map wasn't created (unlikely given prepare_classifier_data runs first)
        num_classes = 4782  # Max from unicode csv

    classifier = KuzushijiClassifier(num_classes=num_classes, pretrained=True).to(
        Config.DEVICE
    )
    cls_optimizer = optim.Adam(classifier.parameters(), lr=Config.CLASSIFIER_LR)

    fit_classifier(
        classifier,
        train_cls_loader,
        val_cls_loader,
        cls_optimizer,
        Config.DEVICE,
        epochs=Config.CLASSIFIER_EPOCHS,
    )

    # ==========================================
    # 4. Full Validation & Metric Calculation
    # ==========================================
    print("\n=== Running Full Validation Inference ===")

    # Load trained models for inference
    det_weights = os.path.join(Config.WORKING_DIR, "detector_best.pth")
    cls_weights = os.path.join(Config.WORKING_DIR, "classifier_best.pth")

    tiled_detector = TiledDetector(det_weights, Config.DEVICE)
    batch_classifier = BatchClassifier(
        cls_weights, Config.CLASS_MAP_PATH, Config.DEVICE
    )

    val_df = pd.read_csv(Config.VAL_METADATA, keep_default_na=False)

    all_preds = []
    all_gts = []

    # For Failure Analysis
    fa_data = []  # {'image_area': int, 'num_gt': int, 'num_pred': int, 'f1': float}

    for i, row in val_df.iterrows():
        image_id = row["image_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        label_str = row["labels"]

        # Parse GT
        gt_parsed = parse_labels(label_str)
        # GT format for metric: (label, x, y, w, h)
        gt_list = [(item[0], item[1], item[2], item[3], item[4]) for item in gt_parsed]
        all_gts.append({"image_id": image_id, "gt": gt_list})

        if not os.path.exists(file_path):
            all_preds.append({"image_id": image_id, "preds": []})
            continue

        # Inference
        import cv2

        image = cv2.imread(file_path)
        if image is None:
            all_preds.append({"image_id": image_id, "preds": []})
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Detect
        detections = tiled_detector.detect(
            image
        )  # returns [x1, y1, x2, y2, score, cls]

        # Classify
        labeled_preds = batch_classifier.classify(image, detections)

        # Format Preds: (label, x, y) -> Center point
        pred_list = []
        for code, box in labeled_preds:
            x1, y1, x2, y2 = box[0:4]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            pred_list.append((code, cx, cy))

        all_preds.append({"image_id": image_id, "preds": pred_list})

        # Failure Analysis Data Collection
        # Calculate per-image F1 roughly for analysis
        # (This is expensive to do exactly per image inside loop, so we approximate or just store counts)
        fa_data.append(
            {
                "image_area": h * w,
                "aspect_ratio": w / h,
                "num_gt": len(gt_list),
                "num_pred": len(pred_list),
                "diff_count": abs(len(gt_list) - len(pred_list)),
            }
        )

        if (i + 1) % 100 == 0:
            print(f"Validated {i+1}/{len(val_df)}")

    # Calculate Metric
    metrics = calc_modified_f1(all_preds, all_gts)
    final_f1 = metrics["f1"]

    print(f"Final Validation Metric: {final_f1}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    fa_df = pd.DataFrame(fa_data)

    if not fa_df.empty:
        # Correlation between error magnitude (count difference) and features
        corr_area = fa_df["diff_count"].corr(fa_df["image_area"])
        corr_ar = fa_df["diff_count"].corr(fa_df["aspect_ratio"])
        corr_density = fa_df["diff_count"].corr(fa_df["num_gt"])

        print(f"Correlation (Error Magnitude vs Image Area): {corr_area:.4f}")
        print(f"Correlation (Error Magnitude vs Aspect Ratio): {corr_ar:.4f}")
        print(f"Correlation (Error Magnitude vs Character Count): {corr_density:.4f}")

        # Identify if we are under-predicting or over-predicting
        total_gt = fa_df["num_gt"].sum()
        total_pred = fa_df["num_pred"].sum()
        print(f"Total GT Characters: {total_gt}")
        print(f"Total Predicted Characters: {total_pred}")
        if total_pred < total_gt:
            print("Systematic Error: Under-prediction (Low Recall)")
        else:
            print("Systematic Error: Over-prediction (Low Precision)")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.8455090517492287

    if final_f1 > THRESHOLD:
        print(
            f"\nValidation metric ({final_f1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            test_metadata_path=Config.TEST_METADATA,
            detector_weights=det_weights,
            classifier_weights=cls_weights,
            output_csv="./submission/submission.csv",
        )
    else:
        print(
            f"\nValidation metric ({final_f1}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
