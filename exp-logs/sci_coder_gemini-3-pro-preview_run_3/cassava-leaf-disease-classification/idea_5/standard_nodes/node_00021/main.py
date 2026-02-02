import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import CassavaClassifier
from library.engine import train_model
from library.inference import generate_submission


def main():
    # ------------------------------------------------------------------
    # 1. Configuration and Setup
    # ------------------------------------------------------------------
    # Override CFG settings for a fast baseline execution
    CFG.epochs = 6

    # Ensure output directory exists (handled by CFG setup, but good practice)
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Set deterministic seeds
    seed_everything(CFG.seed)

    device = CFG.device

    # ------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------
    # Load dataloaders with caching enabled
    train_loader, val_loader, mixup_fn = get_dataloaders(load_cached_data=True)

    # ------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------
    model = CassavaClassifier(
        model_name=CFG.model_name,
        pretrained=True,
        num_classes=CFG.num_classes,
        img_size=CFG.img_size,
    )
    model.to(device)

    # ------------------------------------------------------------------
    # 4. Optimizer and Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    # train_model handles the training loop, validation, and saving the best model
    _ = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device, mixup_fn
    )

    # ------------------------------------------------------------------
    # 6. Final Validation and Failure Analysis
    # ------------------------------------------------------------------
    # Load the best model weights for analysis
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model weights not found.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_targets.extend(targets.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate and print Final Validation Metric
    final_acc = accuracy_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    analysis_df = pd.DataFrame({"target": val_targets, "prediction": val_preds})
    # Error magnitude: 1 if incorrect, 0 if correct
    analysis_df["error"] = (analysis_df["target"] != analysis_df["prediction"]).astype(
        int
    )

    # Calculate correlation between error and class label
    # This identifies if specific classes are systematically more prone to errors
    corr_label = analysis_df["error"].corr(analysis_df["target"])
    print(f"Correlation between Error Magnitude and Target Label: {corr_label}")

    # Print error rate per class for detailed insight
    print("Error Rate per Class:")
    class_errors = analysis_df.groupby("target")["error"].mean()
    print(class_errors)

    # ------------------------------------------------------------------
    # 7. Submission Generation
    # ------------------------------------------------------------------
    THRESHOLD = 0.9022696929238986

    if final_acc > THRESHOLD:
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric {final_acc} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
