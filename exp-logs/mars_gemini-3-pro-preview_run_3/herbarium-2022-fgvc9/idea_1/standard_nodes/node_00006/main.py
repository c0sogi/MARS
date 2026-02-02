import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, compute_metrics
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # We use the full dataset (debug=False) but will limit epochs for speed
    train_loader, val_loader, _ = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
    )

    # 3. Training
    print("Initializing Trainer...")
    trainer = Trainer(device=device)

    # Train for Config.NUM_EPOCHS (8) as per Lesson solution_lesson_node_00004
    print(f"Starting training ({Config.NUM_EPOCHS} epochs)...")
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # 4. Validation & Failure Analysis
    print("Performing validation inference for analysis...")
    trainer.model.eval()

    val_preds = []
    val_labels = []
    val_confs = []

    # Disable gradient calculation for inference to save memory and time
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Use mixed precision for efficiency
            with torch.cuda.amp.autocast():
                outputs = trainer.model(images)
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                # Get max probability (confidence) and predicted class
                confs, preds = torch.max(probs, dim=1)

            val_preds.append(preds.cpu().numpy())
            val_labels.append(labels.numpy())
            val_confs.append(confs.cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(val_preds)
    y_true = np.concatenate(val_labels)
    y_conf = np.concatenate(val_confs)

    # Compute and print the required metric
    final_metric = compute_metrics(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("Running Failure Analysis...")

    # Error mask (1 for incorrect, 0 for correct)
    errors = (y_pred != y_true).astype(int)

    # Analysis 1: Correlation between Error and Prediction Confidence
    # We expect a negative correlation (higher confidence -> lower error)
    if len(errors) > 1:
        corr_conf = np.corrcoef(errors, y_conf)[0, 1]
        print(f"Correlation between Error and Prediction Confidence: {corr_conf}")
    else:
        print("Not enough samples for correlation analysis.")

    # Analysis 2: Correlation between Error and Class Frequency
    # We check if the model struggles more with rare classes (Input Feature: Class Frequency)
    try:
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        class_counts = df_train["category_id"].value_counts().to_dict()

        # Map true labels to their training frequency
        freqs = np.array([class_counts.get(lbl, 0) for lbl in y_true])

        if len(errors) > 1:
            corr_freq = np.corrcoef(errors, freqs)[0, 1]
            print(f"Correlation between Error and Class Frequency: {corr_freq}")
    except Exception as e:
        print(f"Could not perform class frequency analysis: {e}")

    # 5. Submission
    # Only generate submission if metric is strictly higher than the threshold
    threshold = 0.5930838412243743
    if final_metric > threshold:
        print(
            f"Validation metric {final_metric} > {threshold}. Generating submission for test set..."
        )
        # Use the best model saved by the trainer during the fit process
        generate_submission(checkpoint_path=trainer.best_model_path, device=device)
    else:
        print(f"Validation metric {final_metric} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
