import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library functions
from library.utils import seed_everything, save_checkpoint, compute_score
from library.dataset import get_dataloaders
from library.model import RetinopathyModel, train_one_epoch, validate, inference


def analyze_failures(val_csv_path, predictions, targets, root_dir="./input"):
    """
    Performs failure analysis by correlating error magnitude with image meta-features.
    """
    print("\n=== Failure Analysis ===")

    # Load metadata
    df = pd.read_csv(val_csv_path)

    # Ensure alignment: The validate function returns preds/targets in order of the loader.
    # The loader is sequential (shuffle=False).
    # We assume strict alignment between df and loader outputs.

    # Calculate Error Magnitude
    # predictions are continuous scores, targets are integers
    errors = np.abs(np.array(predictions) - np.array(targets))

    # Collect Meta-Features
    widths = []
    heights = []
    file_sizes = []
    intensities = []

    print("Extracting meta-features for validation set...")
    for _, row in df.iterrows():
        full_path = os.path.join(root_dir, row["file_path"])

        # File Size
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))

            # Image Stats
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                intensities.append(img.mean())
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
            "intensity": intensities,
        }
    )

    # Calculate Correlations
    features = ["width", "height", "file_size", "intensity"]
    print("\nCorrelation between Error Magnitude and Input Features:")
    for feat in features:
        if analysis_df[feat].std() > 0:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
            print(f"{feat}: {corr:.6f}")
        else:
            print(f"{feat}: NaN (No variance)")


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 16
    IMAGE_SIZE = 512
    LR = 1e-4
    # Cite solution_lesson_node_00027: Maintain high weight decay (0.05) for ConvNeXt robustness
    WEIGHT_DECAY = 5e-2
    OUTPUT_DIR = "./working/idea_7"
    METADATA_DIR = "./metadata"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Data
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=8,
        metadata_dir=METADATA_DIR,
    )

    # 3. Model
    print("Initializing Model...")
    model = RetinopathyModel(
        model_name="convnext_small.fb_in1k", pretrained=True, num_classes=4
    )
    model = model.to(device)

    # 4. Optimization
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_score = -float("inf")

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(val_loader, model, criterion, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {val_score:.4f}"
        )

        # Save Checkpoint
        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                },
                is_best=True,
                checkpoint_dir=OUTPUT_DIR,
            )

    # 6. Final Evaluation & Failure Analysis
    print("\nTraining Complete. Loading best model for analysis...")
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Re-run validation to get predictions for analysis
    print("Running validation inference...")
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            true_labels = targets.sum(dim=1)

            val_preds.extend(scores.cpu().numpy())
            val_targets.extend(true_labels.cpu().numpy())

    final_qwk = compute_score(val_targets, val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_qwk}")

    # Failure Analysis
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    analyze_failures(val_csv_path, val_preds, val_targets)

    # 7. Conditional Submission
    THRESHOLD = 0.922975135423079

    if final_qwk > THRESHOLD:
        print(
            f"\nValidation score ({final_qwk}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Use the inference function from library.model which handles TTA
        submission_df = inference(test_loader, model, device)

        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nValidation score ({final_qwk}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
