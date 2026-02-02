import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy import stats

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_macro_f1, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import ArcFaceResNet
from library.train import train_one_epoch, validate, generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")
    print(f"Training for {Config.NUM_EPOCHS} epochs.")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading dataloaders...")
    # We must use the full dataset to ensure valid metrics and submission
    dataloaders = get_dataloaders(load_cached_data=True, debug_sample_size=None)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing ArcFaceResNet model...")
    model = ArcFaceResNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    best_f1 = -1.0

    print("Starting training loop...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.6f}"
        )

        # Save Best Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                best_filename=Config.MODEL_SAVE_PATH,
            )
            print(f"  -> New best model saved (F1: {best_f1:.6f})")

    # ---------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Final Evaluation ---")

    # Load the best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    model.eval()

    # Collect all predictions and labels from the validation set
    print("Running full validation inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # labels are needed for metric calculation later, but not for inference forward pass

            # Inference: get cosine similarities (no label passed)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Final Metric
    final_f1 = calculate_macro_f1(all_labels, all_preds)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis: Correlation between Error and Class Frequency
    print("\n--- Failure Analysis ---")
    try:
        # Load training metadata to get class frequencies
        train_df = pd.read_csv(Config.TRAIN_CSV)
        class_counts = train_df["category_id"].value_counts().to_dict()

        # Map validation samples to the frequency of their true class in the training set
        # Use .get(x, 0) to handle potential unseen classes (though unlikely with stratified split)
        val_class_freqs = np.array([class_counts.get(label, 0) for label in all_labels])

        # Determine Error: 1 if prediction is incorrect, 0 if correct
        # We want to see if 'Error' is correlated with 'Frequency'
        errors = (all_preds != all_labels).astype(int)

        # Calculate Point Biserial Correlation
        # (Correlation between a binary variable and a continuous variable)
        if len(np.unique(errors)) > 1:
            corr, p_val = stats.pointbiserialr(errors, val_class_freqs)
            print(
                f"Correlation between Error Magnitude and Class Frequency: {corr:.4f}"
            )
            if corr < 0:
                print(
                    "  -> Negative correlation implies higher class frequency is associated with lower error rate."
                )
            else:
                print(
                    "  -> Positive correlation implies higher class frequency is associated with higher error rate."
                )
        else:
            print(
                "Skipping correlation: All predictions were either correct or incorrect."
            )

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.1780649791722345

    if final_f1 > THRESHOLD:
        print(
            f"\nMetric {final_f1:.6f} > Threshold {THRESHOLD:.6f}. Generating submission..."
        )
        # Use the provided library function to generate submission
        generate_submission(
            Config.MODEL_SAVE_PATH, debug_sample_size=None, load_cached_data=True
        )
    else:
        print(
            f"\nMetric {final_f1:.6f} <= Threshold {THRESHOLD:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
