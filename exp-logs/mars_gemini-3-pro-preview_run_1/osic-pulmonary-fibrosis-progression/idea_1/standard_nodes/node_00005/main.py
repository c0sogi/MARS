import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import MIPLinearDecayNet
from library.engine import Trainer


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = Config.get_device()

    # 2. Load Data
    # Using the provided data loader function.
    # Config.LOAD_CACHED_DATA is True by default in Config, utilizing preprocessed MIPs.
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Initialize Model
    model = MIPLinearDecayNet().to(device)

    # 4. Setup Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Reduce LR if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training
    trainer = Trainer(model, optimizer, device, scheduler)

    # Train for a limited number of epochs (10) to ensure fast execution
    # The dataset is small, so 10 epochs is sufficient for a baseline
    trainer.fit(train_loader, val_loader, epochs=10, patience=3)

    # 6. Evaluation & Metric
    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute validation loss
    val_loss = trainer.evaluate(val_loader)

    # The loss function is defined as -Metric.
    # Therefore, Metric = -Loss.
    final_metric = -val_loss
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing Failure Analysis on Validation Set...")
    model.eval()

    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            weeks = batch["weeks"].to(device)
            true_fvc = batch["fvc_true"].to(device)

            # Forward pass
            pred_slope, pred_conf = model(images, tabular)

            # Calculate predicted FVC
            pred_fvc = base_fvc + (pred_slope.view(-1) * weeks)

            # Calculate absolute error
            abs_error = torch.abs(true_fvc - pred_fvc).cpu().numpy()

            # Extract features for correlation analysis
            # Tabular tensor structure: [Age_Norm, Sex_0, Sex_1, Smoke_0, Smoke_1, Smoke_2, Percent_Norm]
            tab_np = tabular.cpu().numpy()
            weeks_np = weeks.cpu().numpy()
            base_fvc_np = base_fvc.cpu().numpy()
            true_fvc_np = true_fvc.cpu().numpy()

            for i in range(len(abs_error)):
                analysis_data.append(
                    {
                        "Error": abs_error[i],
                        "Weeks": weeks_np[i],
                        "Base_FVC": base_fvc_np[i],
                        "True_FVC": true_fvc_np[i],
                        "Age_Norm": tab_np[i, 0],
                        "Percent_Norm": tab_np[i, -1],
                    }
                )

    df_analysis = pd.DataFrame(analysis_data)

    # Calculate correlations with Error
    if not df_analysis.empty:
        correlations = df_analysis.corr()["Error"].sort_values(ascending=False)
        print("Correlation between Model Error and Features:")
        print(correlations)
    else:
        print("No validation data available for analysis.")

    # 8. Submission Generation
    baseline_metric = -7.496936493589167
    if final_metric > baseline_metric:
        print(
            f"Metric {final_metric:.4f} improved over baseline {baseline_metric:.4f}. Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"Metric {final_metric:.4f} did not improve over baseline {baseline_metric:.4f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
