import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.trainer import Trainer
from library.inference import Predictor
from library.dataset import KuzushijiDataset
from library.utils import (
    load_metadata,
    calc_f1_score,
    parse_ground_truth,
    seed_everything,
)


def main():
    # --- 1. Configuration for Fast Baseline ---
    # Limit epochs to ensure completion within strict time limits while allowing convergence
    Config.EPOCHS = 5
    # Keep batch size conservative to avoid OOM on 1024x1024 inputs
    Config.BATCH_SIZE = 4
    # Use the full dataset (DEBUG=False) to ensure the metric is representative
    Config.DEBUG = False

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print("Initializing Trainer...")
    trainer = Trainer()

    # --- 2. Training ---
    print("Starting Training...")
    trainer.fit()

    # --- 3. Validation & Failure Analysis ---
    print("Starting Failure Analysis...")

    # Load the best model weights for analysis to ensure we analyze the best performing state
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current model weights.")

    trainer.model.eval()

    # Load validation metadata
    val_df = load_metadata(Config.VAL_METADATA_PATH)

    # Create Validation DataLoader
    # We need a custom loader here to access image-level metadata (dimensions) easily in the loop
    val_dataset = KuzushijiDataset(val_df, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    print("Running inference on validation set for analysis...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(Config.DEVICE)
            img_ids = batch["image_id"]
            orig_hs = batch["orig_h"]
            orig_ws = batch["orig_w"]

            # Forward pass
            outputs = trainer.model(imgs)

            # Decode predictions using the trainer's decode method
            detections = trainer._decode(
                outputs["hm"],
                outputs["wh"],
                outputs["reg"],
                outputs["cls"],
                K=Config.MAX_DETECTIONS,
            )
            detections = detections.cpu().numpy()

            # Process each image in batch
            for i, img_id in enumerate(img_ids):
                # Get image dimensions
                h = orig_hs[i].item()
                w = orig_ws[i].item()

                # Get Predictions
                det = detections[i]
                mask = det[:, 2] >= Config.CONF_THRESHOLD
                det = det[mask]

                # Restore coordinates
                scale = Config.IMG_SIZE / max(h, w)
                resized_w = w * scale
                resized_h = h * scale
                pad_w = (Config.IMG_SIZE - resized_w) / 2
                pad_h = (Config.IMG_SIZE - resized_h) / 2

                pred_strs = []
                for d in det:
                    x, y, score, cls_idx = d

                    # Remove padding and rescale
                    x = (x - pad_w) / scale
                    y = (y - pad_h) / scale
                    x = max(0, min(w, x))
                    y = max(0, min(h, y))

                    cls_idx = int(cls_idx)
                    if cls_idx in trainer.idx_to_char:
                        char = trainer.idx_to_char[cls_idx]
                        pred_strs.append(f"{char} {int(x)} {int(y)}")
                pred_str = " ".join(pred_strs)

                # Get Ground Truth
                gt_rows = val_df[val_df["image_id"] == img_id]
                if len(gt_rows) > 0:
                    gt_str = gt_rows.iloc[0]["labels"]
                else:
                    gt_str = ""

                # Calculate Metrics
                # 1. Number of annotations (complexity proxy)
                gt_anns = parse_ground_truth(gt_str)
                num_anns = len(gt_anns)

                # 2. F1 Score for this specific image
                t_df = pd.DataFrame({"image_id": [img_id], "labels": [gt_str]})
                p_df = pd.DataFrame({"image_id": [img_id], "labels": [pred_str]})

                f1 = calc_f1_score(t_df, p_df)

                results.append(
                    {
                        "image_id": img_id,
                        "img_width": w,
                        "img_height": h,
                        "num_anns": num_anns,
                        "f1": f1,
                        "error": 1.0 - f1,  # Error magnitude
                    }
                )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Compute Final Metric
    # We use the best score recorded by the trainer (Global F1) as the official metric
    final_metric = trainer.early_stopping.best_score
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlations
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    if not results_df.empty:
        # Calculate correlation between metadata and error (1 - F1)
        correlations = results_df[
            ["img_width", "img_height", "num_anns", "error"]
        ].corr()["error"]
        print(correlations.drop("error"))
    else:
        print("No validation results to analyze.")

    # --- 4. Submission ---
    THRESHOLD = 0.7679033467456621

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Initialize predictor (loads best_model.pth by default)
        predictor = Predictor(checkpoint_name="best_model.pth")
        predictor.run(output_path="./submission/submission.csv")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
