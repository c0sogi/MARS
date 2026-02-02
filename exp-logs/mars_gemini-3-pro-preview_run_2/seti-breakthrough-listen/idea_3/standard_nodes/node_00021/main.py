import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint, get_score
from library.dataset import SETIDataset, get_transforms
from library.model import SETIModel
from library.engine import train_one_epoch, valid_one_epoch, inference_fn

# Suppress warnings
warnings.filterwarnings("ignore")


def run():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Fast Baseline
    # --------------------------------------------------------------------------
    seed_everything(Config.seed)

    # Override Config for optimized execution
    Config.epochs = 8
    Config.train_subset_size = 30000

    # Create submission directory as per prompt requirements
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    print(f"Device: {Config.device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load Metadata
    df_train = pd.read_csv(Config.train_csv)
    df_val = pd.read_csv(Config.val_csv)
    df_test = pd.read_csv(Config.test_csv)

    # Subsample training data for fast baseline
    if len(df_train) > Config.train_subset_size:
        df_train = df_train.sample(
            n=Config.train_subset_size, random_state=Config.seed
        ).reset_index(drop=True)

    print(f"Training on {len(df_train)} samples")
    print(f"Validating on {len(df_val)} samples")

    # Datasets
    train_dataset = SETIDataset(df_train, transform=get_transforms("train"))
    val_dataset = SETIDataset(df_val, transform=get_transforms("valid"))
    test_dataset = SETIDataset(df_test, transform=get_transforms("test"))

    # DataLoaders
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

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = SETIModel(pretrained=Config.pretrained)
    model.to(Config.device)

    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.T_0,
        T_mult=Config.T_mult,
        eta_min=Config.min_lr,
        last_epoch=-1,
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_score = 0.0
    best_model_path = Config.model_path

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, Config.device, epoch
        )

        # Validate
        val_loss, val_score = valid_one_epoch(model, val_loader, Config.device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_score:.6f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                },
                best_model_path,
            )
            print(f"Saved Best Model at Epoch {epoch+1} with AUC: {best_score:.6f}")

    # --------------------------------------------------------------------------
    # 5. Final Validation and Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Failure Analysis ---")

    # Load best model
    checkpoint = load_checkpoint(best_model_path, Config.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(Config.device)
    model.eval()

    # Get predictions on validation set
    # We need both targets and predictions. valid_one_epoch returns metrics,
    # but we need raw preds for analysis. We'll run a quick inference loop.
    val_preds = []
    val_targets = []

    # Collect basic input features for correlation: Mean, Std, Max
    # Doing this for all val samples might be slow, so we do it for a subset or on the fly.
    # Given the constraints, we will calculate stats for the first 1000 samples.

    analysis_stats = []
    analysis_errors = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(Config.device)

            # Inference
            with torch.cuda.amp.autocast():
                outputs = model(images)
                preds = outputs.sigmoid().cpu().numpy().flatten()

            targets_np = targets.numpy().flatten()

            val_preds.extend(preds)
            val_targets.extend(targets_np)

            # Feature Extraction for Failure Analysis (Limit to first few batches to save time)
            if i < 20:  # Approx 640 samples
                images_np = images.cpu().float().numpy()  # (B, 3, 1638, 256)

                for j in range(images_np.shape[0]):
                    # Calculate error
                    error = abs(targets_np[j] - preds[j])

                    # Calculate input features (using the first channel)
                    # Image is normalized, so these are stats of the normalized signal
                    img_flat = images_np[j, 0, :, :].flatten()
                    mean_val = np.mean(img_flat)
                    std_val = np.std(img_flat)
                    max_val = np.max(img_flat)

                    analysis_stats.append([mean_val, std_val, max_val])
                    analysis_errors.append(error)

    final_val_auc = get_score(np.array(val_targets), np.array(val_preds))
    print(f"Final Validation Metric: {final_val_auc}")

    # Correlation Analysis
    if len(analysis_errors) > 0:
        stats_arr = np.array(analysis_stats)  # (N, 3)
        errors_arr = np.array(analysis_errors)

        # Pearson correlation
        corrs = []
        feat_names = ["Mean", "Std", "Max"]
        for k in range(3):
            if np.std(stats_arr[:, k]) > 1e-9 and np.std(errors_arr) > 1e-9:
                corr = np.corrcoef(stats_arr[:, k], errors_arr)[0, 1]
            else:
                corr = 0.0
            corrs.append(corr)

        print("Correlation between Error Magnitude and Input Features:")
        for name, r in zip(feat_names, corrs):
            print(f"  {name}: {r:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    threshold = 0.8500096914162987

    if final_val_auc > threshold:
        print(
            f"\nValidation score ({final_val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set
        test_preds = inference_fn(model, test_loader, Config.device)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": df_test["id"], "target": test_preds})

        # Save
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation score ({final_val_auc}) did not exceed threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
