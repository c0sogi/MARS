import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, MetricMonitor
from library.dataset import load_dataset_dataframe, ISICDataset, get_transforms
from library.model import ISICModel
from library.engine import fit, evaluate, predict_and_submit


def run_failure_analysis(model, val_loader, df_val, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between prediction error and input features.
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    all_targets = []
    all_probs = []

    # Collect predictions
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate absolute error
    errors = np.abs(all_targets - all_probs)

    # Create analysis dataframe
    df_analysis = df_val.copy()
    # Ensure alignment (assuming loader order matches df order if shuffle=False)
    # The val_loader in main() is created with shuffle=False
    df_analysis["error"] = errors
    df_analysis["prediction"] = all_probs

    # Encode categorical features for correlation analysis
    if "sex" in df_analysis.columns:
        df_analysis["sex_encoded"] = pd.factorize(df_analysis["sex"])[0]
    if "anatom_site_general_challenge" in df_analysis.columns:
        df_analysis["site_encoded"] = pd.factorize(
            df_analysis["anatom_site_general_challenge"]
        )[0]

    # Calculate correlations
    features_to_check = ["age_approx", "sex_encoded", "site_encoded"]
    correlations = {}

    for feat in features_to_check:
        if feat in df_analysis.columns:
            # Handle NaNs in features by filling with median or -1
            series = df_analysis[feat].fillna(-1)
            corr = np.corrcoef(series, df_analysis["error"])[0, 1]
            correlations[feat] = corr

    print("Correlation between Error and Features:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    df_train = load_dataset_dataframe(
        Config.TRAIN_CSV, debug_size=Config.DEBUG_SUBSET_SIZE
    )
    df_val = load_dataset_dataframe(Config.VAL_CSV, debug_size=Config.DEBUG_SUBSET_SIZE)
    df_test = load_dataset_dataframe(
        Config.TEST_CSV, debug_size=Config.DEBUG_SUBSET_SIZE
    )

    print(f"Train size: {len(df_train)}")
    print(f"Val size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")

    # 3. Datasets & Loaders
    train_dataset = ISICDataset(
        df_train, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = ISICDataset(df_val, transforms=get_transforms("val"), mode="val")
    test_dataset = ISICDataset(df_test, transforms=get_transforms("test"), mode="test")

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    model = ISICModel(pretrained=True)
    model = model.to(device)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 6. Training
    print("\nStarting Training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=Config.EPOCHS,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 7. Final Validation & Metrics
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Define criterion for evaluation (same as training)
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_loss, val_auc = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 8. Failure Analysis
    run_failure_analysis(model, val_loader, df_val, device)

    # 9. Submission
    baseline_auc = 0.8860349415712139
    if val_auc > baseline_auc:
        print("\nGenerating Submission...")
        predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not beat baseline ({baseline_auc}). Skipping submission."
        )

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
