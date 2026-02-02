import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import get_datasets
from library.model import SETIModel
from library.engine import SETIEngine


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # Set seeds for reproducibility
    seed_everything(Config.seed)
    device = Config.device

    # Override Config for optimized execution
    Config.epochs = 12

    print(f"Running on device: {device}")
    print(f"Training for {Config.epochs} epochs")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # We use debug=False to train on the full dataset to achieve the required AUC.
    train_dataset, val_dataset, test_dataset = get_datasets(debug=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = SETIModel(pretrained=True)
    model.to(device)

    # -------------------------------------------------------------------------
    # 4. Optimizer and Scheduler
    # -------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    engine = SETIEngine(model, device, optimizer, scheduler)
    print("Starting training...")
    engine.train(train_loader, val_loader, epochs=Config.epochs)

    # -------------------------------------------------------------------------
    # 6. Evaluation and Failure Analysis
    # -------------------------------------------------------------------------
    print("Loading best model for final evaluation...")
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    model.eval()

    val_preds = []
    val_targets = []
    meta_stats = []

    print("Running validation inference and feature extraction...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Inference
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets_np)

            # Feature Extraction for Failure Analysis
            # Move images to CPU for numpy statistics
            # images shape: (B, 3, H, W). We use the first channel as they are identical/stacked.
            imgs_np = images.cpu().numpy()

            for i in range(imgs_np.shape[0]):
                img_data = imgs_np[i, 0, :, :]
                meta_stats.append(
                    {
                        "mean": np.mean(img_data),
                        "std": np.std(img_data),
                        "max": np.max(img_data),
                        "min": np.min(img_data),
                    }
                )

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate and Print Final Metric
    final_auc = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    df_analysis = pd.DataFrame(meta_stats)
    df_analysis["error"] = errors

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.8382845148797015

    if final_auc > threshold:
        print(
            f"Validation Metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        # engine.predict handles loading the best model and TTA
        engine.predict(test_loader)
    else:
        print(
            f"Validation Metric ({final_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
