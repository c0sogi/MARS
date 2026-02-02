import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import calculate_kuzushiji_metrics, format_prediction_string
from library.dataset import KuzushijiDataset, get_transforms, collate_fn
from library.model import get_kuzushiji_model
from library.engine import train_kuzushiji_model, inference, set_seeds


def main():
    # 1. Setup & Configuration
    set_seeds(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Constants
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.005
    THRESHOLD = 0.8041786589915787
    WORKING_DIR = "./working"
    SUBMISSION_PATH = "./submission/submission.csv"

    # 2. Data Preparation
    print("Initializing Datasets and DataLoaders...")
    train_ds = KuzushijiDataset(mode="train", transforms=get_transforms(train=True))
    val_ds = KuzushijiDataset(mode="val", transforms=get_transforms(train=False))
    test_ds = KuzushijiDataset(mode="test", transforms=get_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model Construction
    # Number of classes = number of unique characters + 1 for background
    num_classes = len(train_ds.char_to_int) + 1
    print(f"Initializing Cascade R-CNN with {num_classes} classes...")
    model = get_kuzushiji_model(num_classes)
    model.to(device)

    # 4. Training
    print("Configuring Optimizer...")
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    # train_kuzushiji_model handles the loop, validation, scheduler, and saving best model
    best_f1 = train_kuzushiji_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        num_epochs=NUM_EPOCHS,
        patience=5,
        save_dir=WORKING_DIR,
    )

    # 5. Validation & Failure Analysis
    print("Loading best model for final evaluation and failure analysis...")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print("Running inference on validation set...")
    int_to_char = val_ds.int_to_char
    int_to_image_id = {v: k for k, v in val_ds.image_id_map.items()}

    pred_results = []
    img_features = {}  # Store image_id -> {area, num_chars_gt}

    # Iterate over validation set to get predictions and extract features
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, (target, output) in enumerate(zip(targets, outputs)):
                img_int_id = target["image_id"].item()
                img_str_id = int_to_image_id.get(img_int_id, str(img_int_id))

                # Extract Input Feature: Image Area (Original)
                w_orig = target["orig_w"].item()
                h_orig = target["orig_h"].item()
                area = float(w_orig * h_orig)
                img_features[img_str_id] = area

                # Format Predictions
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Rescale boxes to original dimensions
                _, h_curr, w_curr = images[i].shape
                scale_x = w_orig / w_curr
                scale_y = h_orig / h_curr

                boxes[:, 0] *= scale_x
                boxes[:, 2] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 3] *= scale_y

                label_str = format_prediction_string(boxes, labels, int_to_char)
                pred_results.append({"image_id": img_str_id, "labels": label_str})

    pred_df = pd.DataFrame(pred_results)

    # Calculate Final Validation Metric
    metrics = calculate_kuzushiji_metrics(val_ds.df, pred_df)
    final_metric = metrics["f1"]
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Input Features
    print("Performing failure analysis...")
    f1_scores = []
    areas = []
    char_counts = []

    # Calculate F1 per image
    for _, row in val_ds.df.iterrows():
        img_id = row["image_id"]

        # Get GT char count
        gt_labels = row.get("labels", "")
        if pd.isna(gt_labels) or not gt_labels:
            n_chars = 0
        else:
            # Format is U X Y W H, so 5 parts per char
            n_chars = len(gt_labels.split()) // 5

        # Get Prediction for this image
        pred_row = pred_df[pred_df["image_id"] == img_id]
        mini_pred = (
            pred_row
            if not pred_row.empty
            else pd.DataFrame({"image_id": [img_id], "labels": [""]})
        )
        mini_gt = pd.DataFrame([row])

        # Calculate metric for single image
        m = calculate_kuzushiji_metrics(mini_gt, mini_pred)

        f1_scores.append(m["f1"])
        char_counts.append(n_chars)
        areas.append(img_features.get(img_id, 0))

    # Error Magnitude = 1 - F1
    errors = 1.0 - np.array(f1_scores)

    # Calculate Correlations
    if len(errors) > 1:
        corr_chars = np.corrcoef(errors, char_counts)[0, 1]
        corr_area = np.corrcoef(errors, areas)[0, 1]
    else:
        corr_chars = 0.0
        corr_area = 0.0

    print(f"Correlation (Error vs Num Characters): {corr_chars:.4f}")
    print(f"Correlation (Error vs Image Area): {corr_area:.4f}")

    # 6. Submission
    if final_metric > THRESHOLD:
        print(
            f"Validation metric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        inference(model, test_loader, device, output_path=SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
