import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import LungDataset
from library.model import DualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.engine import Engine


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to find correlations
    between error magnitude and input features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()

    all_errors = []
    all_metas = []  # To store [Base_FVC, Delta_Week]
    all_tabular = []  # To store tabular inputs if accessible, or we rely on metadata

    # We need to access the original dataframe features for correlation.
    # Since the loader returns processed tensors, we'll collect indices or
    # rely on the fact that the loader is sequential if shuffle=False.
    # However, for robustness, we will extract features from the batch tensors
    # (reversing normalization where possible or using normalized values directly).

    feature_data = {
        "Age_Norm": [],
        "Percent_Norm": [],
        "Delta_Week": [],
        "Base_FVC": [],
        "True_FVC": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch)
            targets = batch["target"]
            meta = batch["meta"]  # [Base_FVC, Delta_Week]
            tabular = batch["tabular"]  # [Age_Norm, Pct_Norm, Sex..., Smoke...]

            # Reconstruct Predictions
            alpha = outputs[:, 0]
            base_fvc = meta[:, 0]
            dt = meta[:, 1]

            pred_fvc = base_fvc + alpha * dt

            # Calculate Absolute Error
            error = torch.abs(targets - pred_fvc).cpu().numpy()
            all_errors.extend(error)

            # Collect Features for Correlation
            # tabular[0] is Age_Norm, tabular[1] is Pct_Norm
            feature_data["Age_Norm"].extend(tabular[:, 0].cpu().numpy())
            feature_data["Percent_Norm"].extend(tabular[:, 1].cpu().numpy())
            feature_data["Delta_Week"].extend(dt.cpu().numpy())
            feature_data["Base_FVC"].extend(base_fvc.cpu().numpy())
            feature_data["True_FVC"].extend(targets.cpu().numpy())

    # Convert to arrays
    errors = np.array(all_errors)

    print(f"Mean Absolute Error on Validation Set: {np.mean(errors):.4f}")

    # Calculate Correlations
    print("\nCorrelation between Absolute Error and Features:")
    for feat_name, feat_vals in feature_data.items():
        feat_vals = np.array(feat_vals)
        if len(np.unique(feat_vals)) > 1:
            corr, _ = pearsonr(feat_vals, errors)
            print(f"  {feat_name}: {corr:.4f}")
        else:
            print(f"  {feat_name}: N/A (Constant value)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = LungDataset(mode="train")
    val_dataset = LungDataset(mode="val")

    # Use num_workers from Config, pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DualAxisNet()

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 5. Loss Function
    criterion = LaplaceLogLikelihoodLoss()

    # 6. Training Engine
    engine = Engine(model, optimizer, device=device, scheduler=scheduler)

    # 7. Run Training
    print("Starting Training...")
    engine.fit(
        train_loader,
        val_loader,
        criterion,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    # 8. Final Evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")
    load_checkpoint(best_model_path, model, device=device)

    # Compute Final Metric
    final_metric = engine.evaluate(val_loader, criterion, epoch="Final")
    print(f"Final Validation Metric: {final_metric}")

    # 9. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 10. Submission Logic
    # Threshold from instructions: -6.510164260864258
    # Note: Metric is negative, so higher is better (e.g. -6.0 > -6.5)
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = LungDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        engine.predict(test_loader)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
