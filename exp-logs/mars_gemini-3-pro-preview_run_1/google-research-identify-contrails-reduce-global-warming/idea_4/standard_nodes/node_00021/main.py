import os
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.train import train_model
from library.model import HRNetSegmenter
from library.dataset import get_dataloader
from library.utils import rle_encode, set_seed


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override configuration for a fast but effective baseline run
    Config.EPOCHS = 6
    Config.BATCH_SIZE = 32

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # Execute the training loop provided in the library
    # This will save the best model to ./working/idea_4/checkpoints/best_model.pth
    train_model(debug=False)

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    # Load the best model for evaluation
    model = HRNetSegmenter(pretrained=False)
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    val_loader = get_dataloader(
        mode="validation", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Accumulators for Global Dice
    intersection_sum = 0.0
    union_sum = 0.0
    smooth = 1e-6
    threshold = 0.5

    # Accumulators for Failure Analysis
    record_ids = []
    sample_errors = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            r_ids = batch["record_id"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Flatten for Global Dice calculation
            pred_flat = preds.view(-1)
            true_flat = masks.view(-1).float()

            intersection_sum += (pred_flat * true_flat).sum().item()
            union_sum += pred_flat.sum().item() + true_flat.sum().item()

            # Per-sample analysis for failure correlation
            # Iterate through the batch
            for i in range(len(images)):
                p_flat = preds[i].view(-1)
                t_flat = masks[i].view(-1).float()

                inter = (p_flat * t_flat).sum().item()
                union = p_flat.sum().item() + t_flat.sum().item()

                dice = (2.0 * inter + smooth) / (union + smooth)
                error = 1.0 - dice

                record_ids.append(str(r_ids[i]))
                sample_errors.append(error)

    # Compute and print Final Metric
    final_metric = (2.0 * intersection_sum + smooth) / (union_sum + smooth)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    # Load validation metadata to correlate errors with physical parameters
    val_meta = pd.read_csv(Config.VALIDATION_CSV)
    val_meta["record_id"] = val_meta["record_id"].astype(str)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"record_id": record_ids, "error": sample_errors})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_meta, on="record_id", how="left")

    print("Failure Analysis (Correlation with Error):")
    features_to_analyze = ["timestamp", "row_min", "col_min"]

    for feat in features_to_analyze:
        if feat in analysis_df.columns:
            # Drop NaNs to ensure valid correlation calculation
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr = valid_data[feat].corr(valid_data["error"])
                print(f"{feat}: {corr}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.5973177358563411

    if final_metric > THRESHOLD_SCORE:
        test_loader = get_dataloader(
            mode="test", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )

        sub_record_ids = []
        sub_encoded_pixels = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                r_ids = batch["record_id"]

                logits = model(images)
                probs = torch.sigmoid(logits)

                # Convert to numpy for RLE encoding
                preds_np = (probs > threshold).float().cpu().numpy()

                for i in range(len(images)):
                    # preds_np[i] shape is (1, H, W)
                    mask = preds_np[i, 0, :, :]
                    encoded = rle_encode(mask)

                    sub_record_ids.append(r_ids[i])
                    sub_encoded_pixels.append(encoded)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"record_id": sub_record_ids, "encoded_pixels": sub_encoded_pixels}
        )

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
