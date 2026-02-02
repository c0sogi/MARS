"""
Main execution script for Speech Command Recognition.
Orchestrates training, validation, failure analysis, and submission.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules are accessible
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.train import Trainer
from library.predict import generate_submission


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print("Initializing Configuration...")

    # Adjust configuration for a fast but effective baseline
    # 20 epochs ensures full convergence for EfficientNet
    Config.NUM_EPOCHS = 20
    Config.BATCH_SIZE = 128
    # Optimize data loading for the available 12 vCPUs
    Config.NUM_WORKERS = 8

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading Datasets...")
    # Loaders handle the splitting and preprocessing defined in library.dataset
    loaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Starting Training Pipeline...")
    trainer = Trainer(
        learning_rate=Config.LEARNING_RATE,
        num_epochs=Config.NUM_EPOCHS,
        device=Config.DEVICE,
    )

    # Train the model with early stopping
    trainer.fit(train_loader, val_loader, patience=5)

    # ==========================================
    # 4. Validation Assessment
    # ==========================================
    print("\n=== Validation Assessment ===")

    # Load the best model saved during training for evaluation
    best_model_path = Config.BEST_MODEL_PATH
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found!")
        return

    checkpoint = load_checkpoint(
        trainer.model, path=best_model_path, device=Config.DEVICE
    )
    if checkpoint:
        print(
            f"Loaded best model from Epoch {checkpoint.get('epoch')} (Val Acc: {checkpoint.get('val_acc'):.6f})"
        )

    trainer.model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_energies = []

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            outputs = trainer.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Collect data for metrics
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Calculate input feature: Signal Energy (Mean of spectrogram)
            # This serves as the input feature for failure analysis
            # inputs shape: (B, 1, F, T)
            batch_energies = inputs.mean(dim=(1, 2, 3)).cpu().numpy()
            all_energies.extend(batch_energies)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_energies = np.array(all_energies)

    # Calculate Final Metric (Multiclass Accuracy)
    correct = all_preds == all_targets
    final_accuracy = np.mean(correct)

    # PRINT REQUIRED METRIC WITH FULL PRECISION
    print(f"Final Validation Metric: {final_accuracy}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")

    # 5.1 Error Magnitude Calculation
    # Error Magnitude is defined as (1.0 - Probability of the True Class)
    # This quantifies how "wrong" or "uncertain" the model was about the correct label.
    rows = np.arange(len(all_targets))
    prob_true_class = all_probs[rows, all_targets]
    error_magnitude = 1.0 - prob_true_class

    # 5.2 Correlation with Input Feature (Signal Energy)
    # We analyze if the signal energy (loudness/background level) correlates with model error.
    if len(all_energies) > 1:
        correlation_matrix = np.corrcoef(all_energies, error_magnitude)
        correlation = correlation_matrix[0, 1]
        print(
            f"Correlation between Signal Energy and Error Magnitude: {correlation:.6f}"
        )

        if abs(correlation) > 0.1:
            print(
                "-> Observation: Weak to moderate correlation detected between signal energy and error."
            )
        else:
            print(
                "-> Observation: No significant correlation detected between signal energy and error."
            )

    # 5.3 Class-wise Analysis
    # Identify which commands are most problematic
    print("\nClass-wise Error Rates:")
    df_analysis = pd.DataFrame(
        {"target": all_targets, "correct": correct, "error_mag": error_magnitude}
    )

    class_stats = df_analysis.groupby("target").agg(
        {"correct": "mean", "error_mag": "mean"}
    )
    class_stats["error_rate"] = 1.0 - class_stats["correct"]

    # Print stats for each class
    for label_id, row in class_stats.iterrows():
        label_name = Config.ID2LABEL[label_id]
        print(
            f"  {label_name:<10}: Error Rate = {row['error_rate']:.4f} | Mean Error Mag = {row['error_mag']:.4f}"
        )

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.9853666694539677

    if final_accuracy > THRESHOLD:
        print(
            f"\nValidation Accuracy ({final_accuracy:.16f}) exceeds threshold ({THRESHOLD})."
        )
        print("Generating Submission...")
        # Use the predict module to generate submission from the best checkpoint
        generate_submission(batch_size=Config.BATCH_SIZE, device=Config.DEVICE)
    else:
        print(
            f"\nValidation Accuracy ({final_accuracy:.16f}) does NOT exceed threshold ({THRESHOLD})."
        )
        print("Skipping Submission Generation.")


if __name__ == "__main__":
    main()
