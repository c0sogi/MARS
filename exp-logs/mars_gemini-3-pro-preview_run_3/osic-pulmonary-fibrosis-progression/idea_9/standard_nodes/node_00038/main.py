import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.train import Trainer
from library.utils import load_checkpoint


def run_failure_analysis(trainer, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    print("\n=== Failure Analysis ===")
    trainer.model.eval()
    device = Config.DEVICE

    data_records = []

    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch: img, tabular, t_rel, target, pid, current_week
            imgs, tab, t_rel, target, _, current_week = batch

            imgs = imgs.to(device)
            tab_dev = tab.to(device)
            t_rel_dev = t_rel.to(device)

            # Forward pass
            mu_norm, sigma_norm = trainer.model(imgs, tab_dev, t_rel_dev)

            # Inverse Transformation
            mu = (
                mu_norm.cpu().numpy().flatten() * Config.TARGET_STD + Config.TARGET_MEAN
            )
            target_raw = (
                target.numpy().flatten() * Config.TARGET_STD + Config.TARGET_MEAN
            )

            # Calculate Error
            error = np.abs(target_raw - mu)

            # Extract features from tabular tensor (CPU)
            # tab structure: [Base_FVC_Norm, Age_Norm, Sex, Smoke]
            tab_cpu = tab.numpy()
            base_fvc_norm = tab_cpu[:, 0]
            age_norm = tab_cpu[:, 1]
            sex = tab_cpu[:, 2]
            smoke = tab_cpu[:, 3]

            # Revert normalization for interpretability
            base_fvc = base_fvc_norm * Config.TARGET_STD + Config.TARGET_MEAN
            age = age_norm * 6.62 + 67.58  # Using constants from data.py

            weeks = current_week.numpy().flatten()

            for i in range(len(error)):
                data_records.append(
                    {
                        "Error": error[i],
                        "Age": age[i],
                        "Sex": sex[i],
                        "Smoking": smoke[i],
                        "Base_FVC": base_fvc[i],
                        "Weeks": weeks[i],
                    }
                )

    df_analysis = pd.DataFrame(data_records)

    # Calculate correlations
    features = ["Age", "Sex", "Smoking", "Base_FVC", "Weeks"]
    print("Correlation between Absolute Error and Features:")
    for feat in features:
        if feat in df_analysis.columns:
            corr, _ = stats.pearsonr(df_analysis["Error"], df_analysis[feat])
            print(f"  {feat}: {corr:.6f}")


def main():
    # 1. Setup
    Config.setup()
    # Cite solution_lesson_node_00027: Fine-tuning requires more epochs
    Config.EPOCHS = 50

    print(f"Initializing run with {Config.EPOCHS} epochs...")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Training
    print("Starting training...")
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # 4. Load Best Model for Evaluation
    best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        print(f"Loading best model from {best_path}...")
        load_checkpoint(best_path, trainer.model, device=Config.DEVICE)
    else:
        print("Warning: No checkpoint found. Using last model state.")

    # 5. Final Validation Metric
    print("Computing final validation metric...")
    val_score = trainer.validate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    run_failure_analysis(trainer, val_loader)

    # 7. Submission
    threshold = -6.57744688338769
    if val_score > threshold:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nValidation score ({val_score}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
