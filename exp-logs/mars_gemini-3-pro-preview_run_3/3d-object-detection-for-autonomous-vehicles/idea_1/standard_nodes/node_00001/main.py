import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import warnings

# Import provided library modules
from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PointPillars
from library.train import train_model, calculate_image_iou_metric, set_seed
from library.inference import decode_predictions, nms_process, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # Ensure reproducibility
    set_seed(Config.SEED)

    # ==============================================================================
    # 1. Training Phase
    # ==============================================================================
    # We train the model using a subset of the data (5000 samples) and limited epochs (3)
    # to ensure the script completes quickly (Fast Baseline).
    # train_model handles the training loop, validation on the subset, and saving the best model.
    print(">>> Starting Training Phase...")
    train_model(epochs=3, batch_size=Config.BATCH_SIZE, max_samples=5000)

    # ==============================================================================
    # 2. Final Validation & Failure Analysis
    # ==============================================================================
    # We must validate on the ENTIRE hold-out validation set to report the final metric.
    # We also collect data per sample to perform failure analysis.

    print("\n>>> Starting Final Validation and Failure Analysis...")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = PointPillars().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize the Full Validation Dataset
    val_dataset = NuScenesDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
    )

    # Prepare resources for metric calculation
    anchors = torch.from_numpy(val_dataset.anchors).to(device)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Create a lookup map for ground truth labels
    token_to_label = dict(
        zip(val_dataset.metadata["sample_token"], val_dataset.metadata["label"])
    )

    # Variables for aggregation
    analysis_records = []
    total_metric = 0.0
    num_samples = 0

    with torch.no_grad():
        # Iterate over the validation set
        for batch in val_loader:
            # Move batch to device
            pillars = batch["pillars"].to(device)
            coors = batch["coors"].to(device)
            n_points = batch["n_points"].to(device)
            sample_tokens = batch["sample_tokens"]

            input_dict = {
                "pillars": pillars,
                "coors": coors,
                "n_points": n_points,
                "sample_tokens": sample_tokens,
            }

            # Inference
            output = model(input_dict)
            cls_preds = output["cls_preds"]
            reg_preds = output["reg_preds"]

            # Decode raw predictions into boxes
            batch_boxes, batch_scores = decode_predictions(
                cls_preds, reg_preds, anchors
            )

            # Process each sample in the batch individually
            for i, token in enumerate(sample_tokens):
                # 1. Apply NMS to get final predictions for this sample
                boxes = batch_boxes[i]
                scores = batch_scores[i]

                final_boxes, final_scores, _ = nms_process(
                    boxes,
                    scores,
                    score_thresh=Config.NMS_SCORE_THRESHOLD,
                    iou_thresh=Config.NMS_IOU_THRESHOLD,
                    max_proposals=Config.MAX_PROPOSALS,
                )

                if final_boxes is None:
                    final_boxes = torch.empty((0, 7), device=device)
                    final_scores = torch.empty((0,), device=device)

                # 2. Retrieve Ground Truth
                label_str = token_to_label.get(token, "")
                gt_boxes_np = val_dataset._parse_labels(label_str)

                if len(gt_boxes_np) > 0:
                    gt_boxes = torch.from_numpy(gt_boxes_np[:, :7]).to(device)
                else:
                    gt_boxes = torch.empty((0, 7), device=device)

                # 3. Calculate Metric (mAP across thresholds)
                score = calculate_image_iou_metric(
                    final_boxes, final_scores, gt_boxes, thresholds
                )
                total_metric += score
                num_samples += 1

                # 4. Collect Data for Failure Analysis
                # Feature A: Number of Ground Truth Objects
                num_gt = len(gt_boxes)

                # Feature B: Number of Lidar Points in the sample
                # 'coors' has shape (N_pillars, 4) where column 0 is the batch index.
                # 'n_points' has shape (N_pillars, ) containing point count per pillar.
                # We sum the points in all pillars corresponding to the current batch index 'i'.
                mask = coors[:, 0] == i
                num_lidar_points = n_points[mask].sum().item()

                analysis_records.append(
                    {
                        "score": score,
                        "error": 1.0 - score,
                        "num_gt": num_gt,
                        "num_points": num_lidar_points,
                    }
                )

    # Report Final Metric
    final_val_metric = total_metric / num_samples if num_samples > 0 else 0.0
    print(f"Final Validation Metric: {final_val_metric}")

    # Report Failure Analysis
    if analysis_records:
        df = pd.DataFrame(analysis_records)

        # Calculate Pearson correlation between Error and Features
        corr_gt = df["error"].corr(df["num_gt"])
        corr_points = df["error"].corr(df["num_points"])

        print("Failure Analysis Correlations:")
        print(f"Correlation (Error vs Num GT Objects): {corr_gt}")
        print(f"Correlation (Error vs Num Lidar Points): {corr_points}")

    # ==============================================================================
    # 3. Submission Generation
    # ==============================================================================
    print("\n>>> Generating Submission...")
    # Generate predictions for the test set and save to submission.csv
    generate_submission(
        model_path=Config.MODEL_SAVE_PATH, output_path=Config.SUBMISSION_PATH
    )


if __name__ == "__main__":
    main()
