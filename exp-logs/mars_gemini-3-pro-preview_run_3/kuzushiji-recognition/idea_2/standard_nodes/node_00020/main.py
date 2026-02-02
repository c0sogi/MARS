import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    collate_fn_detector,
    collate_fn_classifier,
    decode_detections,
    f1_score_calc,
)
from library.models import CenterNetDetector, CharacterClassifier
from library.dataset import (
    KuzushijiDetectorDataset,
    KuzushijiCropDataset,
    get_transforms,
    get_class_map,
)
from library.preprocess import prepare_classifier_data
from library.engine import train_detector, train_classifier
from library.inference import InferencePipeline


def validate_pipeline():
    """
    Runs the full detection + classification pipeline on the validation set
    to compute the official metric and perform failure analysis.
    """
    device = Config.DEVICE
    print("Loading models for validation...")

    # Load Detector
    detector = CenterNetDetector(pretrained=False)
    detector.load_state_dict(
        torch.load(Config.DETECTOR_MODEL_PATH, map_location=device)
    )
    detector.to(device).eval()

    # Load Classifier
    classifier = CharacterClassifier(pretrained=False)
    classifier.load_state_dict(
        torch.load(Config.CLASSIFIER_MODEL_PATH, map_location=device)
    )
    classifier.to(device).eval()

    # Load Class Map
    _, idx_to_char = get_class_map(load_cached=True)

    # Load Validation Metadata
    df_val = pd.read_csv(Config.VAL_METADATA_PATH, keep_default_na=False)

    # Transforms
    det_transform = get_transforms("detector", "test", Config.DETECTOR_IMG_SIZE)
    cls_transform = get_transforms("classifier", "test", Config.CLASSIFIER_IMG_SIZE)

    all_preds = []
    all_targets = []
    img_stats = []

    print(f"Validating on {len(df_val)} images...")

    for idx, row in df_val.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        labels_str = row["labels"]

        # Parse Targets
        targets = []
        if labels_str:
            parts = labels_str.split()
            for i in range(len(parts) // 5):
                try:
                    code = parts[i * 5]
                    x = int(parts[i * 5 + 1])
                    y = int(parts[i * 5 + 2])
                    w = int(parts[i * 5 + 3])
                    h = int(parts[i * 5 + 4])
                    targets.append({"label": code, "x": x, "y": y, "w": w, "h": h})
                except ValueError:
                    continue
        all_targets.append(targets)

        # Load Image
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            all_preds.append([])
            img_stats.append(
                {"num_chars": len(targets), "f1": 0.0, "area": 0, "aspect_ratio": 0}
            )
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # 1. Detection
        # Apply transform (Resize + Pad)
        # We pass dummy bbox/labels to satisfy Albumentations interface
        t_det = det_transform(image=img_rgb, bboxes=[[0, 0, 10, 10]], labels=[1])
        img_tensor = t_det["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = detector(img_tensor)
            # Decode
            detections = decode_detections(
                out["heatmap"],
                out["size_map"],
                out["offset_map"],
                K=Config.MAX_PREDICTIONS,
            )
            detections = detections.cpu().numpy()[0]

        # Filter by confidence
        mask = detections[:, 4] >= Config.CONF_THRESHOLD
        valid_dets = detections[mask]

        img_preds = []

        if len(valid_dets) > 0:
            # Correct Coordinates (Model Input -> Original Image)
            input_size = Config.DETECTOR_IMG_SIZE
            scale = input_size / max(orig_h, orig_w)
            pad_top = (input_size - int(orig_h * scale)) // 2
            pad_left = (input_size - int(orig_w * scale)) // 2

            # In-place modification of valid_dets
            # x_center
            valid_dets[:, 0] = (valid_dets[:, 0] - pad_left) / scale
            # y_center
            valid_dets[:, 1] = (valid_dets[:, 1] - pad_top) / scale
            # width
            valid_dets[:, 2] = valid_dets[:, 2] / scale
            # height
            valid_dets[:, 3] = valid_dets[:, 3] / scale

            # 2. Classification
            crops = []
            valid_indices = []

            for i, det in enumerate(valid_dets):
                cx, cy, w, h = det[0], det[1], det[2], det[3]

                # Convert center to top-left/bottom-right
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                # Clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)

                # Handle degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    x1 = max(0, int(cx - 16))
                    y1 = max(0, int(cy - 16))
                    x2 = min(orig_w, int(cx + 16))
                    y2 = min(orig_h, int(cy + 16))

                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                t_cls = cls_transform(image=crop)
                crops.append(t_cls["image"])
                valid_indices.append(i)

            if crops:
                crops_tensor = torch.stack(crops).to(device)

                # Batch inference
                cls_logits = []
                bs = 256
                with torch.no_grad():
                    for k in range(0, len(crops_tensor), bs):
                        chunk = crops_tensor[k : k + bs]
                        cls_logits.append(classifier(chunk))

                    cls_logits = torch.cat(cls_logits, dim=0)
                    pred_cls_idx = torch.argmax(cls_logits, dim=1).cpu().numpy()

                # Assemble predictions
                for k, idx in enumerate(pred_cls_idx):
                    det_idx = valid_indices[k]
                    det = valid_dets[det_idx]
                    label = idx_to_char.get(idx, "U+0000")
                    score = float(det[4])

                    img_preds.append(
                        {
                            "label": label,
                            "x": det[0],  # Center X
                            "y": det[1],  # Center Y
                            "score": score,
                        }
                    )

        all_preds.append(img_preds)

        # Stats for failure analysis
        metrics = f1_score_calc([img_preds], [targets])
        img_stats.append(
            {
                "num_chars": len(targets),
                "f1": metrics["f1"],
                "area": orig_w * orig_h,
                "aspect_ratio": orig_w / (orig_h + 1e-6),
            }
        )

    # Compute Final Metric
    final_metrics = f1_score_calc(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metrics['f1']}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    df_stats = pd.DataFrame(img_stats)
    df_stats["error"] = 1.0 - df_stats["f1"]

    if len(df_stats) > 1:
        corr_chars = df_stats["error"].corr(df_stats["num_chars"])
        corr_area = df_stats["error"].corr(df_stats["area"])
        corr_ar = df_stats["error"].corr(df_stats["aspect_ratio"])

        print(f"Correlation (Error vs Num Chars): {corr_chars:.4f}")
        print(f"Correlation (Error vs Image Area): {corr_area:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")
    else:
        print("Not enough data for correlation analysis.")

    return final_metrics["f1"]


def main():
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # Override Epochs for Fast Baseline
    # Increased epochs for deeper backbone and harder augmentation
    Config.DETECTOR_EPOCHS = 20
    Config.CLASSIFIER_EPOCHS = 10

    # 2. Preprocessing
    print("\n=== Step 1: Preprocessing ===")
    prepare_classifier_data(split="train", load_cached=True)
    prepare_classifier_data(split="val", load_cached=True)

    # 3. Train Detector
    print("\n=== Step 2: Training Detector ===")
    train_ds = KuzushijiDetectorDataset(split="train")
    val_ds = KuzushijiDetectorDataset(split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_detector,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_detector,
        pin_memory=True,
    )

    detector = CenterNetDetector(pretrained=True).to(device)
    optimizer_det = torch.optim.AdamW(detector.parameters(), lr=Config.DETECTOR_LR)

    train_detector(detector, train_loader, val_loader, optimizer_det, device)

    # Cleanup to save VRAM
    del detector, optimizer_det, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()

    # 4. Train Classifier
    print("\n=== Step 3: Training Classifier ===")
    train_crop_ds = KuzushijiCropDataset(split="train")
    val_crop_ds = KuzushijiCropDataset(split="val")

    train_crop_loader = DataLoader(
        train_crop_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_classifier,
        pin_memory=True,
    )
    val_crop_loader = DataLoader(
        val_crop_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn_classifier,
        pin_memory=True,
    )

    classifier = CharacterClassifier(pretrained=True).to(device)
    optimizer_cls = torch.optim.AdamW(classifier.parameters(), lr=Config.CLASSIFIER_LR)

    train_classifier(
        classifier, train_crop_loader, val_crop_loader, optimizer_cls, device
    )

    # Cleanup
    del (
        classifier,
        optimizer_cls,
        train_crop_loader,
        val_crop_loader,
        train_crop_ds,
        val_crop_ds,
    )
    torch.cuda.empty_cache()

    # 5. Validation & Failure Analysis
    print("\n=== Step 4: Validation & Failure Analysis ===")
    val_metric = validate_pipeline()

    # 6. Submission
    print("\n=== Step 5: Submission ===")
    threshold = 0.8253955574017564

    if val_metric > threshold:
        print(f"Validation metric {val_metric} > {threshold}. Generating submission...")
        pipeline = InferencePipeline(device=device)
        pipeline.run()
    else:
        print(f"Validation metric {val_metric} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
