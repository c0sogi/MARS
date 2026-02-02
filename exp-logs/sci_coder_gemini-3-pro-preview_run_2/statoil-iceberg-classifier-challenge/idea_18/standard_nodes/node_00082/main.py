import os
import numpy as np
import pandas as pd
import torch
from library.utils import seed_everything, get_logger
from library.model import WB_DSN
from library.data_loader import (
    process_data,
    get_folds,
    get_dataloaders,
    get_test_loader,
)
from library.trainer import ModelTrainer


def main():
    # 1. Setup & Initialization
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("RunFile")

    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Loading and processing data...")
    train_data, test_data = process_data(load_cached_data=True)

    # 3. Cross-Validation Loop
    n_folds = 5
    folds = get_folds(train_data, n_splits=n_folds, seed=42)

    cv_scores = []
    trained_models = []

    # Containers for Failure Analysis (aggregating across folds)
    all_val_preds = []
    all_val_targets = []
    all_val_angles = []
    all_val_b1_means = []
    all_val_b2_means = []

    logger.info(f"Starting {n_folds}-Fold Cross-Validation...")

    for fold_idx in range(n_folds):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Get DataLoaders
        train_loader, val_loader = get_dataloaders(
            fold_idx, folds, train_data, batch_size=32
        )

        # Initialize Model and Trainer
        model = WB_DSN().to(device)
        trainer = ModelTrainer(model, device, logger=logger, learning_rate=2e-4)

        # Train
        # Using 80 epochs with extended patience for "Low and Slow" convergence (Cite solution_lesson_node_00023)
        best_loss = trainer.fit(train_loader, val_loader, epochs=80, patience=20)
        cv_scores.append(best_loss)
        trained_models.append(model)

        # Inference on Validation Set for Analysis
        # We manually run inference to extract features for correlation analysis
        model.eval()
        fold_preds = []
        fold_targets = []
        fold_angles = []
        fold_b1_means = []
        fold_b2_means = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                imgs, angles = inputs
                imgs = imgs.to(device)
                angles = angles.to(device)

                # Forward pass
                outputs = model(imgs, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                # Collect predictions and targets
                fold_preds.extend(probs)
                fold_targets.extend(targets.numpy().flatten())
                fold_angles.extend(angles.cpu().numpy().flatten())

                # Collect image stats (Mean of Band 1 and Band 2)
                # imgs shape: (B, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
                # Calculate mean over spatial dims (2, 3)
                b1_m = imgs[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
                b2_m = imgs[:, 1, :, :].mean(dim=(1, 2)).cpu().numpy()

                fold_b1_means.extend(b1_m)
                fold_b2_means.extend(b2_m)

        # Aggregate
        all_val_preds.extend(fold_preds)
        all_val_targets.extend(fold_targets)
        all_val_angles.extend(fold_angles)
        all_val_b1_means.extend(fold_b1_means)
        all_val_b2_means.extend(fold_b2_means)

    # 4. Validation Reporting
    final_metric = np.mean(cv_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("\nPerforming Failure Analysis...")
    y_true = np.array(all_val_targets)
    y_pred = np.array(all_val_preds)
    errors = np.abs(y_true - y_pred)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": all_val_angles,
            "b1_mean": all_val_b1_means,
            "b2_mean": all_val_b2_means,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = 0.16676861786296204
    if final_metric < threshold:
        logger.info(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_loader = get_test_loader(test_data, batch_size=32)
        ensemble_preds = np.zeros(len(test_data["ids"]))

        # Ensemble Inference
        with torch.no_grad():
            for i, model in enumerate(trained_models):
                model.eval()
                fold_test_preds = []
                for inputs in test_loader:
                    imgs, angles = inputs
                    imgs = imgs.to(device)
                    angles = angles.to(device)

                    outputs = model(imgs, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_test_preds.extend(probs)

                ensemble_preds += np.array(fold_test_preds)

        # Average predictions
        ensemble_preds /= n_folds

        # Save Submission
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"

        sub_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": ensemble_preds})

        # Ensure correct column order and format
        sub_df = sub_df[["id", "is_iceberg"]]
        sub_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.info(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
