import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, collate_fn, box_cxcywh_to_xyxy
from library.dataset import SIIMDataset
from library.transforms import get_transforms
from library.model import MultiTaskDeformableDETR
from library.loss import SetCriterion
from library.engine import train_one_epoch, evaluate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    # We use 5 epochs to ensure completion within 2 hours while learning enough features.
    Config.EPOCHS = 5

    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Loading
    print("Loading Datasets...")
    # Train
    train_dataset = SIIMDataset(split="train", transform=get_transforms("train"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    # Validation
    val_dataset = SIIMDataset(split="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = MultiTaskDeformableDETR().to(device)
    criterion = SetCriterion().to(device)

    # Optimizer (Separate LR for backbone and transformer)
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": Config.LR_TRANSFORMER,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": Config.LR_BACKBONE,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=Config.LR_TRANSFORMER, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    print("Starting Training...")
    best_map = 0.0

    for epoch in range(Config.EPOCHS):
        # Train
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            max_norm=Config.CLIP_MAX_NORM,
        )

        # Validate
        val_stats = evaluate(model, criterion, val_loader, device)
        current_map = val_stats["map"]

        print(
            f"Epoch {epoch} Summary: Train Loss: {train_stats['loss']:.4f} | Val mAP: {current_map:.4f}"
        )

        # Save Best Model
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with mAP: {best_map:.4f}")

    # 5. Final Evaluation & Metric Reporting
    print("\n--- Training Complete. Loading Best Model for Final Evaluation ---")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Re-run evaluation to confirm metric and ensure state is correct
    final_stats = evaluate(model, criterion, val_loader, device)
    final_metric = final_stats["map"]

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Performing Failure Analysis ---")
    # We analyze correlation between Loss (Error Magnitude) and Metadata
    analysis_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    analysis_data = []
    criterion.eval()

    with torch.no_grad():
        for samples, targets in tqdm(analysis_loader, desc="Analyzing Failures"):
            samples = samples.to(device)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            total_loss = loss_dict["loss"].item()

            # Extract metadata
            # targets[0]['orig_size'] is [h, w]
            h, w = targets[0]["orig_size"].cpu().numpy()
            num_boxes = len(targets[0]["boxes"])

            analysis_data.append(
                {"loss": total_loss, "height": h, "width": w, "num_boxes": num_boxes}
            )

    df_analysis = pd.DataFrame(analysis_data)

    print("\nCorrelations with Error Magnitude (Loss):")
    for feature in ["width", "height", "num_boxes"]:
        corr = df_analysis["loss"].corr(df_analysis[feature])
        print(f"  Loss vs {feature}: {corr:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.1573485090997539

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        test_dataset = SIIMDataset(split="test", transform=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Batch size 1 simplifies ID mapping logic
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=Config.PIN_MEMORY,
        )

        submission_rows = []
        processed_studies = set()

        model.eval()
        with torch.no_grad():
            for samples, targets in tqdm(test_loader, desc="Inference"):
                samples = samples.to(device)
                # targets in test set only contain 'image_id', 'study_id', 'orig_size'

                outputs = model(samples)

                # --- Process Study Prediction ---
                # logits: (1, 4)
                study_logits = outputs["pred_study_logits"][0]
                study_probs = torch.softmax(study_logits, dim=0)
                study_pred_idx = torch.argmax(study_probs).item()
                study_conf = study_probs[study_pred_idx].item()
                study_label_str = Config.ID2STUDY_LABEL[study_pred_idx]

                # Map label string to submission format (e.g., "Negative for Pneumonia" -> "negative")
                # The sample submission uses lowercase first word usually, but let's check task description.
                # Task says: "negative", "typical", "indeterminate", "atypical"
                label_map = {
                    "Negative for Pneumonia": "negative",
                    "Typical Appearance": "typical",
                    "Indeterminate Appearance": "indeterminate",
                    "Atypical Appearance": "atypical",
                }
                study_pred_str = (
                    f"{label_map[study_label_str]} {study_conf:.6f} 0 0 1 1"
                )

                study_id = targets[0]["study_id"]
                image_id = targets[0]["image_id"]

                # Add study row if not already added
                if study_id not in processed_studies:
                    submission_rows.append(
                        {"id": f"{study_id}_study", "PredictionString": study_pred_str}
                    )
                    processed_studies.add(study_id)

                # --- Process Image Prediction ---
                # Check for "Negative" override
                if (
                    Config.FORCE_NONE_ON_NEGATIVE
                    and study_label_str == "Negative for Pneumonia"
                ):
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    # Process Object Queries
                    # logits: (1, Q, 2) -> class 0 is opacity
                    pred_logits = outputs["pred_logits"][0]
                    pred_boxes = outputs["pred_boxes"][0]  # (Q, 4) normalized cxcywh

                    # Get probabilities for 'opacity' class (index 0)
                    probs = pred_logits[:, 0].sigmoid()

                    # Filter by threshold
                    keep = probs > Config.POST_PROCESS_CONF_THRESH

                    if keep.sum() == 0:
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        valid_probs = probs[keep]
                        valid_boxes = pred_boxes[keep]

                        # Convert boxes to absolute xyxy
                        # Get original image size
                        h_orig, w_orig = targets[0]["orig_size"].cpu().numpy()

                        # Note: The model input was resized to Config.IMG_SIZE with padding (Letterbox).
                        # We need to map normalized boxes back to the ORIGINAL image coordinates.
                        # Letterbox logic:
                        # scale = IMG_SIZE / max(h, w)
                        # pad_x = (IMG_SIZE - w*scale) / 2
                        # pad_y = (IMG_SIZE - h*scale) / 2

                        img_size = Config.IMG_SIZE
                        scale = img_size / max(h_orig, w_orig)
                        pad_x = (img_size - w_orig * scale) // 2
                        pad_y = (img_size - h_orig * scale) // 2

                        # Convert normalized (0-1) to letterboxed coordinates (0-IMG_SIZE)
                        boxes_xyxy_lb = box_cxcywh_to_xyxy(valid_boxes) * img_size

                        # Remove padding
                        boxes_xyxy_lb[:, [0, 2]] -= pad_x
                        boxes_xyxy_lb[:, [1, 3]] -= pad_y

                        # Rescale to original size
                        boxes_xyxy_orig = boxes_xyxy_lb / scale

                        # Clip to image boundaries
                        boxes_xyxy_orig[:, [0, 2]] = boxes_xyxy_orig[:, [0, 2]].clamp(
                            min=0, max=w_orig
                        )
                        boxes_xyxy_orig[:, [1, 3]] = boxes_xyxy_orig[:, [1, 3]].clamp(
                            min=0, max=h_orig
                        )

                        # Format string
                        pred_parts = []
                        for i in range(len(valid_probs)):
                            p = valid_probs[i].item()
                            b = boxes_xyxy_orig[i].tolist()
                            pred_parts.append(
                                f"opacity {p:.6f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                            )

                        image_pred_str = " ".join(pred_parts)

                submission_rows.append(
                    {"id": f"{image_id}_image", "PredictionString": image_pred_str}
                )

        # Save Submission
        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} rows.")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
