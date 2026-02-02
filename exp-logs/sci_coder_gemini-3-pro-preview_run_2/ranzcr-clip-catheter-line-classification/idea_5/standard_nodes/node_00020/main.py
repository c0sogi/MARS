import os
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from scipy.stats import pearsonr

from library.config import Config
from library.data import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.engine import ModelEMA, train_one_epoch, validate
from library.utils import seed_everything


def run_pipeline():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    print(f"Starting pipeline on device: {device}")

    # 2. Data Loading & Subsampling
    print("Loading metadata...")
    train_df = pd.read_csv(Config.train_metadata)
    val_df = pd.read_csv(Config.val_metadata)
    test_df = pd.read_csv(Config.test_metadata)

    # Subsample training data for fast baseline (limit to 5000 samples)
    # This ensures the training completes well within the time limit.
    if len(train_df) > 5000:
        print(f"Subsampling training data from {len(train_df)} to 5000 samples.")
        train_df = train_df.sample(n=5000, random_state=Config.seed).reset_index(
            drop=True
        )

    # Initialize Datasets
    train_dataset = CatheterDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = CatheterDataset(
        val_df, transforms=get_transforms("valid"), mode="valid"
    )
    test_dataset = CatheterDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # Initialize DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CatheterModel(pretrained=Config.pretrained).to(device)

    # EMA Setup
    ema_model = None
    if Config.use_ema:
        ema_model = ModelEMA(model, decay=Config.ema_decay, device=device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 4. Training Loop
    print("Starting training...")
    best_auc = 0.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch + 1, ema_model
        )

        # Validate (use EMA model if available)
        eval_model = ema_model.ema if ema_model else model
        val_loss, val_auc = validate(eval_model, val_loader, device)

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(eval_model.state_dict(), best_model_path)

    print(f"Final Validation Metric: {best_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    errors = []
    feature_means = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            # Calculate loss per sample (mean over classes) to represent error magnitude
            loss = criterion(logits, labels).mean(dim=1)
            errors.extend(loss.cpu().numpy())

            # Calculate input feature: Mean pixel intensity of the processed image
            # images: (B, C, H, W) -> mean over spatial and channel dims
            means = images.mean(dim=(1, 2, 3))
            feature_means.extend(means.cpu().numpy())

    # Calculate correlation
    if len(errors) > 0 and len(feature_means) > 0:
        corr, _ = pearsonr(errors, feature_means)
        print(f"Correlation between Error Magnitude and Input Mean Intensity: {corr}")

    # 6. Submission
    threshold = 0.9563622421530574
    if best_auc > threshold:
        print(
            f"\nValidation metric ({best_auc}) meets threshold ({threshold}). Generating submission..."
        )

        preds_all = []
        ids_all = []

        # Inference on Test Set
        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)

                preds_all.append(probs.cpu().numpy())
                ids_all.extend(ids)

        if preds_all:
            preds_arr = np.concatenate(preds_all, axis=0)

            # Create DataFrame
            sub_df = pd.DataFrame(preds_arr, columns=Config.target_cols)
            sub_df.insert(0, "StudyInstanceUID", ids_all)

            # Save
            os.makedirs("./submission", exist_ok=True)
            sub_df.to_csv("./submission/submission.csv", index=False)
            print("Submission saved to ./submission/submission.csv")
    else:
        print(
            f"\nValidation metric {best_auc} is below threshold {threshold}. No submission generated."
        )


if __name__ == "__main__":
    run_pipeline()
