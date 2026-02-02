import pandas as pd
import torch
import numpy as np
import os
from library.config import Config
from library.utils import seed_everything
from library.model import get_species_classifier
from library.data_loader import (
    get_dataloaders,
    get_test_dataloader,
    CroppedSpeciesDataset,
    get_transforms,
    load_detector_bboxes,
)
from library.engine import train_model, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using full dataset for best performance.
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(limit_data=None)

    # 3. Model Initialization
    print("Initializing Model...")
    model = get_species_classifier(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 4. Training
    print("Starting Training...")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        scheduler=scheduler,
        epochs=Config.EPOCHS,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 5. Full Validation Assessment
    print("\n--- Starting Full Validation Assessment ---")

    # Load the best saved model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load full validation metadata and bounding boxes
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    bbox_dict = load_detector_bboxes()

    # Split into High Confidence (Model) and Low Confidence (Heuristic)
    high_conf_mask = val_df["max_detection_conf"] >= Config.CONF_THRESHOLD
    val_high = val_df[high_conf_mask].copy()
    val_low = val_df[~high_conf_mask].copy()

    # Heuristic Prediction: Low confidence images are 'Empty' (Class 0)
    val_low["predicted"] = Config.EMPTY_CLASS_ID

    # Model Prediction: High confidence images
    # We use is_test=True to get image_ids returned from the dataset for accurate mapping
    val_high_dataset = CroppedSpeciesDataset(
        val_high, bbox_dict, transform=get_transforms("val"), is_test=True
    )

    val_high_loader = torch.utils.data.DataLoader(
        val_high_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    high_preds_map = {}

    print(f"Running inference on {len(val_high)} high-confidence validation images...")
    with torch.no_grad():
        for images, image_ids in val_high_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()
            for img_id, pred in zip(image_ids, preds):
                high_preds_map[str(img_id)] = int(pred)

    # Map predictions back to the DataFrame
    val_high["predicted"] = val_high["image_id"].astype(str).map(high_preds_map)

    # Combine results
    val_full = pd.concat([val_high, val_low])

    # Calculate Metric (Categorization Accuracy)
    correct_predictions = (val_full["category_id"] == val_full["predicted"]).sum()
    total_samples = len(val_full)
    accuracy = correct_predictions / total_samples

    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error (0 for correct, 1 for incorrect)
    val_full["error"] = (val_full["category_id"] != val_full["predicted"]).astype(int)

    # Calculate correlation between Error and Max Detection Confidence
    # We fill NaN confidence with 0 just in case, though metadata shouldn't have NaNs based on analysis
    val_full["max_detection_conf"] = val_full["max_detection_conf"].fillna(0.0)

    correlation = val_full["error"].corr(val_full["max_detection_conf"])
    print(f"Correlation between Error and Max Detection Confidence: {correlation:.10f}")

    # 7. Submission
    baseline_metric = 0.6811115261932814
    if accuracy > baseline_metric:
        print(
            f"\n--- Validation Metric ({accuracy:.5f}) > Baseline ({baseline_metric:.5f}). Generating Submission ---"
        )
        test_loader, low_conf_test_df = get_test_dataloader()
        generate_submission(
            model,
            test_loader,
            low_conf_test_df,
            device,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\n--- Validation Metric ({accuracy:.5f}) <= Baseline ({baseline_metric:.5f}). Skipping Submission ---"
        )


if __name__ == "__main__":
    main()
