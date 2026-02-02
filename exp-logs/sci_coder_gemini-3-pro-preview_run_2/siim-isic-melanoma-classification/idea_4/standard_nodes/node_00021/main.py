import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import library components
from library.config import Config
from library.utils import seed_everything, get_roc_auc
from library.dataset import process_metadata, ISICDataset, get_transforms
from library.model import HybridEfficientNet
from library.engine import train_one_epoch, evaluate, predict


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading and processing metadata...")
    train_data, val_data, test_data = process_metadata(
        load_cached_data=True, debug=Config.DEBUG
    )

    # Create Datasets
    train_dataset = ISICDataset(
        image_paths=train_data["image_paths"],
        meta_features=train_data["meta_features"],
        targets=train_data["targets"],
        aux_targets=train_data["aux_targets"],
        transform=get_transforms(data="train"),
    )

    val_dataset = ISICDataset(
        image_paths=val_data["image_paths"],
        meta_features=val_data["meta_features"],
        targets=val_data["targets"],
        aux_targets=val_data["aux_targets"],
        transform=get_transforms(data="valid"),
    )

    test_dataset = ISICDataset(
        image_paths=test_data["image_paths"],
        meta_features=test_data["meta_features"],
        targets=None,
        aux_targets=None,
        transform=get_transforms(data="test"),
    )

    # Create DataLoaders
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

    # 3. Model Initialization
    meta_dim = train_data["meta_features"].shape[1]
    print(f"Initializing HybridEfficientNet with meta_dim={meta_dim}...")
    model = HybridEfficientNet(meta_dim=meta_dim, pretrained=True)
    model.to(device)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Warmup -> Cosine Annealing
    num_epochs = Config.EPOCHS
    warmup_epochs = Config.WARMUP_EPOCHS

    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=Config.ETA_MIN
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )

    # 5. Training Loop
    print(f"Starting training for {num_epochs} epochs on {device}...")
    best_auc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Save Best Model
        if val_auc > best_auc:
            print(
                f"Validation AUC improved ({best_auc:.4f} -> {val_auc:.4f}). Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Calculate final metric on full validation set
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["target"].to(device)

            logits_mal, _ = model(images, meta)
            preds = torch.sigmoid(logits_mal)

            val_preds.extend(preds.cpu().numpy().ravel())
            val_targets.extend(targets.cpu().numpy().ravel())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    final_auc = get_roc_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Load raw validation metadata for correlation analysis
    df_val = pd.read_csv(Config.VAL_META)

    # Align dataframes
    if len(df_val) == len(errors):
        df_val["error"] = errors

        # Correlation with Age
        if "age_approx" in df_val.columns:
            corr_age = df_val["age_approx"].corr(df_val["error"])
            print(f"Correlation (Error vs Age): {corr_age:.4f}")

        # Correlation with Sex
        if "sex" in df_val.columns:
            # Simple encoding for correlation
            df_val["sex_code"] = df_val["sex"].astype("category").cat.codes
            corr_sex = df_val["sex_code"].corr(df_val["error"])
            print(f"Correlation (Error vs Sex): {corr_sex:.4f}")

        # Correlation with Anatom Site
        if "anatom_site_general_challenge" in df_val.columns:
            df_val["site_code"] = (
                df_val["anatom_site_general_challenge"].astype("category").cat.codes
            )
            corr_site = df_val["site_code"].corr(df_val["error"])
            print(f"Correlation (Error vs Anatom Site): {corr_site:.4f}")
    else:
        print("Error: Validation set size mismatch. Skipping specific correlations.")

    # 8. Submission
    THRESHOLD = 0.8887170910773826

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predict(model, test_loader, device)

        # Load test metadata for image names
        df_test = pd.read_csv(Config.TEST_META)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"image_name": df_test["image_name"], "target": test_preds}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    run()
