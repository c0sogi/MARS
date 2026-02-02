import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import CadenceDataset
from library.model import SiameseEfficientNet
from library.engine import train_one_epoch, validate, generate_submission, EarlyStopping
from library.utils import seed_everything


def run_failure_analysis(model, data_loader, device):
    """
    Performs failure analysis on the validation set by correlating
    prediction error with input signal statistics.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    meta_features = []
    errors = []

    with torch.no_grad():
        for batch in data_loader:
            on_input = batch["on_input"].to(device)
            off_input = batch["off_input"].to(device)
            target = batch["target"].to(device)

            # Inference
            logits = model(on_input, off_input)
            probs = torch.sigmoid(logits).squeeze()

            # Calculate Absolute Error
            batch_errors = torch.abs(target - probs).cpu().numpy()
            errors.extend(batch_errors)

            # Extract Meta-Features from the normalized input tensors
            # Flatten spatial dimensions (B, C, H, W) -> (B, C, H*W)
            on_flat = on_input.view(on_input.size(0), -1)
            off_flat = off_input.view(off_input.size(0), -1)

            # Compute stats per sample across all channels/pixels
            mean_on = torch.mean(on_flat, dim=1).cpu().numpy()
            std_on = torch.std(on_flat, dim=1).cpu().numpy()
            max_on = torch.max(on_flat, dim=1).values.cpu().numpy()
            mean_off = torch.mean(off_flat, dim=1).cpu().numpy()

            for i in range(len(mean_on)):
                meta_features.append(
                    {
                        "mean_on": mean_on[i],
                        "std_on": std_on[i],
                        "max_on": max_on[i],
                        "mean_off": mean_off[i],
                        "mean_diff": mean_on[i] - mean_off[i],
                    }
                )

    df_features = pd.DataFrame(meta_features)
    df_features["error"] = errors

    # Calculate correlation
    correlations = (
        df_features.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # 2. Data Preparation
    # Subsample training data to ensures quick baseline execution (< 2 hours)
    MAX_TRAIN_SAMPLES = 15000
    print(f"Loading training metadata from {Config.TRAIN_CSV}...")
    train_df = pd.read_csv(Config.TRAIN_CSV)

    if len(train_df) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_df)} to {MAX_TRAIN_SAMPLES}..."
        )
        # Stratified subsampling
        train_df = (
            train_df.groupby("target", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), int(MAX_TRAIN_SAMPLES * len(x) / len(train_df))),
                    random_state=Config.SEED,
                )
            )
            .reset_index(drop=True)
        )

    # Save subsampled metadata to working directory
    temp_train_path = os.path.join(Config.WORK_DIR, "train_subset.csv")
    train_df.to_csv(temp_train_path, index=False)

    # Initialize Datasets
    train_dataset = CadenceDataset(temp_train_path, mode="train")
    val_dataset = CadenceDataset(Config.VAL_CSV, mode="val")

    # Initialize DataLoaders
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
    print("Initializing SiameseEfficientNet...")
    model = SiameseEfficientNet(pretrained=True)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Early Stopping
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    early_stopping = EarlyStopping(patience=5, mode="max", save_path=best_model_path)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss, val_auc = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Load Best Model
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 6. Final Validation Metric
    print("Performing final validation on full validation set...")
    _, final_auc = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission Generation
    THRESHOLD = 0.5196359687502365
    if final_auc > THRESHOLD:
        print(
            f"Validation metric ({final_auc}) passed threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = CadenceDataset(Config.TEST_CSV, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric ({final_auc}) did not pass threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
