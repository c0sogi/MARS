import os
import pandas as pd
import numpy as np
import torch
import scipy.stats
from library.config import Config
from library.trainer import Trainer
from library.inference import generate_submission

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# Adjust epochs to ensure completion within strict time limits while allowing convergence.
# EfficientNetV2-S on A100 is fast; 5 epochs on ~180k images takes < 30 mins.
Config.EPOCHS = 5
Config.BATCH_SIZE = 64
Config.NUM_WORKERS = 12


def perform_failure_analysis(trainer, val_loader):
    """
    Analyzes model performance on the validation set to identify error patterns.
    Specifically looks at the correlation between error rate and class frequency.
    """
    print("\n=== Starting Failure Analysis ===")

    # 1. Load Training Metadata to calculate class frequencies
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    class_counts = train_df["category_id"].value_counts().to_dict()

    # 2. Run Inference on Validation Set to get per-sample results
    trainer.model.eval()
    all_targets = []
    all_preds = []

    # Use the device from the trainer
    device = trainer.device

    print("Collecting validation predictions...")
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device, non_blocking=True)

            # Forward pass
            if trainer.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = trainer.model(images)
            else:
                outputs = trainer.model(images)

            # Get Top-1 Prediction
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 3. Create Analysis DataFrame
    analysis_df = pd.DataFrame({"target": all_targets, "prediction": all_preds})

    # Calculate Error (1 if wrong, 0 if correct)
    analysis_df["is_error"] = (
        analysis_df["target"] != analysis_df["prediction"]
    ).astype(int)

    # Map Class Frequency (Input Feature)
    # We map the frequency of the *target* class (ground truth)
    analysis_df["class_frequency"] = analysis_df["target"].map(class_counts)

    # Handle classes in validation that might not be in train (though split strategy prevents this mostly)
    analysis_df["class_frequency"] = analysis_df["class_frequency"].fillna(0)

    # 4. Calculate Correlation
    # We correlate 'is_error' (binary) with 'class_frequency' (continuous/ordinal)
    # Point-biserial correlation is appropriate here.
    if len(analysis_df["is_error"].unique()) > 1:
        correlation, p_value = scipy.stats.pointbiserialr(
            analysis_df["is_error"], analysis_df["class_frequency"]
        )
        print(
            f"Correlation between Error and Class Frequency: {correlation:.4f} (p-value: {p_value:.4e})"
        )

        if correlation < 0:
            print(
                "Observation: Negative correlation implies rare classes (lower frequency) have higher error rates."
            )
        else:
            print(
                "Observation: Positive correlation implies frequent classes have higher error rates."
            )
    else:
        print(
            "Could not calculate correlation (all predictions were correct or all were wrong)."
        )

    print("=== Failure Analysis Complete ===\n")


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("Initializing Pipeline...")

    # 1. Initialize Trainer
    trainer = Trainer()

    # 2. Load Data
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = trainer.get_dataloaders(debug=False)

    # 3. Train Model
    print("Starting Training...")
    trainer.fit(train_loader, val_loader)

    # 4. Final Validation Evaluation
    print("Reloading best model for final evaluation...")
    if os.path.exists(Config.MODEL_CHECKPOINT):
        state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)
    else:
        print("Warning: No checkpoint found. Using current weights.")

    print("Calculating Final Validation Metric...")
    val_loss, val_acc1, val_acc5 = trainer.validate(val_loader)

    # Metric is Top-1 Classification Error
    top1_error = 100.0 - val_acc1

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {top1_error}")

    # 5. Failure Analysis
    perform_failure_analysis(trainer, val_loader)

    # 6. Generate Submission
    if top1_error < 26.0:
        print(f"Validation Error ({top1_error:.4f}) < 26.0. Generating Submission...")
        # We call the imported function which handles the test loader and prediction loop internally
        # Note: generate_submission creates a new Trainer instance internally.
        # To ensure it uses the trained model, we rely on the saved checkpoint at Config.MODEL_CHECKPOINT.
        generate_submission(debug=False)
    else:
        print(f"Validation Error ({top1_error:.4f}) >= 26.0. Skipping Submission.")

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
