import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_optimizer_params
from library.dataset import get_datasets
from library.model import HybridSwiGLUNet
from library.engine import fit, evaluate, generate_submission


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set by calculating the correlation
    between prediction error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_targets = []
    all_preds = []
    all_continuous = []

    # Collect predictions and inputs
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device).view(-1, 1)

            outputs = model(continuous, categorical)
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            all_continuous.append(continuous.cpu().numpy())

    targets_np = np.concatenate(all_targets).flatten()
    preds_np = np.concatenate(all_preds).flatten()
    continuous_np = np.concatenate(all_continuous, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(targets_np - preds_np)

    print(f"Mean Absolute Error: {np.mean(errors):.6f}")

    # Calculate correlation with continuous features
    # continuous_np shape: (N, 30)
    # We iterate through the 30 features (f_00 to f_30 excluding f_27)
    # The dataset loader logic maps them to indices 0..29

    # Reconstruct feature names based on dataset logic (f_00..f_30 sans f_27)
    feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    correlations = []
    for i, feat_name in enumerate(feature_names):
        feat_values = continuous_np[:, i]
        # Pearson correlation
        if np.std(feat_values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")
    print("------------------------\n")


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load datasets using cached data if available
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    # Create DataLoaders
    # Pin memory enables faster data transfer to CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # 3. Model Initialization
    model = HybridSwiGLUNet()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer_grouped_parameters = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training
    # We use the fit function which handles the loop, validation, and saving best model
    print("Starting training...")
    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=5,  # Early stopping patience
    )

    # 6. Final Evaluation
    # Reload the best model state is handled inside fit(), but let's be explicit
    # that the model object now contains the best weights.

    # Calculate final metric on the full validation set
    val_loss, final_val_auc = evaluate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission Generation
    # Threshold check
    THRESHOLD = 0.9972883264620234

    if final_val_auc > THRESHOLD:
        print(
            f"Validation AUC ({final_val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # We need the test IDs. The dataset object has them via the metadata/cache logic,
        # but the dataset __getitem__ doesn't return IDs.
        # We can retrieve them from the cached file or metadata.
        # The cleanest way given the library structure is to read the test metadata again
        # or rely on the order preserved by the dataloader (which matches test_metadata.csv).

        test_meta_df = pd.read_csv(Config.TEST_METADATA)
        test_ids = test_meta_df["id"].values

        generate_submission(model, test_loader, test_ids, device)
    else:
        print(
            f"Validation AUC ({final_val_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
