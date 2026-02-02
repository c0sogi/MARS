import os
import cv2
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.configuration import Config
from library.utilities import seed_everything, map5
from library.data_loader import get_dataloaders
from library.architecture import WhaleArcFaceModel
from library.engine import run_training


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlation between error magnitude and image metadata.
    """
    model.eval()

    # 1. Get Per-Sample Scores
    all_scores = []
    all_indices = []

    # We need to access the original dataframe to get file paths for metadata
    # val_loader.dataset is the WhaleDataset
    val_df = val_loader.dataset.df

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # TTA Inference
            logits_orig = model(images, labels=None)
            logits_flip = model(torch.flip(images, dims=[3]), labels=None)
            avg_logits = (logits_orig + logits_flip) / 2.0

            # Top 5 preds
            _, preds = torch.topk(avg_logits, 5, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Calculate score for each sample in batch
            for i in range(len(labels_np)):
                p = preds_np[i].tolist()
                t = labels_np[i]

                score = 0.0
                if t in p:
                    rank = p.index(t)
                    score = 1.0 / (rank + 1)

                all_scores.append(score)
                # Track original index to map back to dataframe
                # batch_idx * batch_size + i works if shuffle=False
                global_idx = batch_idx * val_loader.batch_size + i
                all_indices.append(global_idx)

    # 2. Extract Metadata (Width, Height, Intensity)
    # We iterate through the dataframe in the same order
    meta_stats = []

    print("Performing failure analysis on validation set...")

    # Ensure we align scores with the dataframe
    # val_loader is shuffle=False, so order matches df
    scores_aligned = all_scores[: len(val_df)]

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])

        # Read image for stats
        img = cv2.imread(full_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        # Simple mean intensity
        intensity = np.mean(img) / 255.0

        # Error = 1 - Score
        error = 1.0 - scores_aligned[idx]

        meta_stats.append(
            {
                "Width": w,
                "Height": h,
                "AspectRatio": w / h if h > 0 else 0,
                "MeanIntensity": intensity,
                "Error": error,
            }
        )

    # 3. Compute Correlations
    if len(meta_stats) > 0:
        df_stats = pd.DataFrame(meta_stats)
        correlations = df_stats.corr()["Error"].drop("Error")

        print("-" * 40)
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)
        print("-" * 40)
    else:
        print("Could not compute failure analysis stats.")


def generate_submission(model, test_loader, idx_to_class, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    model.eval()
    results = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # TTA Inference
            logits_orig = model(images, labels=None)
            logits_flip = model(torch.flip(images, dims=[3]), labels=None)
            avg_logits = (logits_orig + logits_flip) / 2.0

            # Top 5 predictions
            _, top5_indices = torch.topk(avg_logits, 5, dim=1)
            top5_indices = top5_indices.cpu().numpy()

            for i, filename in enumerate(filenames):
                # Convert indices to class names
                pred_classes = [idx_to_class[idx] for idx in top5_indices[i]]
                pred_str = " ".join(pred_classes)
                results.append({"Image": filename, "Id": pred_str})

    # Save to CSV
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    # 2. Data Loading
    # load_cached_data=True allows using the classes.npy generated previously
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        load_cached_data=True
    )

    # Update Config with actual number of classes found
    Config.num_classes = num_classes

    # Load class mapping for submission decoding
    # We can reconstruct idx_to_class from the dataset's class_to_idx
    # train_loader.dataset is a WhaleDataset
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Data loaded. Num classes: {num_classes}")

    # 3. Model Initialization
    model = WhaleArcFaceModel(num_classes=num_classes)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 5. Training
    best_score = run_training(
        model, train_loader, val_loader, optimizer, scheduler, device, Config.epochs
    )

    # 6. Final Validation & Analysis
    print("Loading best model for analysis...")
    checkpoint_path = os.path.join(Config.checkpoint_dir, "model_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Recalculate metric on full validation set to be precise and print required format
    # (Although run_training returns best_score, we run it again to ensure state consistency for analysis)
    # Using the eval_fn logic inside run_training, but we can just use the returned best_score
    # if we trust it matches the saved checkpoint. To be safe and explicit:

    # We perform the validation inference manually here to print the exact metric required
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # TTA
            logits = (model(images) + model(torch.flip(images, [3]))) / 2.0
            _, preds = torch.topk(logits, 5, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    final_metric = map5(all_preds, all_targets)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission
    threshold = 0.6306356245
    if final_metric > threshold:
        generate_submission(model, test_loader, idx_to_class, device)
    else:
        print(
            f"Validation metric {final_metric} did not exceed threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
