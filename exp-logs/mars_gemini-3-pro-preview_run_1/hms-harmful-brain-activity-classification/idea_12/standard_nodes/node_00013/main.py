import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence_loss
from library.dataset import HMSDataset
from library.model import AsymmetricCoordinateNet
from library.engine import train_model, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Define probability columns for target extraction
    PROB_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # 2. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Subsample training data for fast baseline execution
    # Using 15,000 samples ensures we finish well within the time limit while having enough data to learn.
    train_df = train_df.sample(
        n=min(15000, len(train_df)), random_state=Config.SEED
    ).reset_index(drop=True)
    print(
        f"Training on {len(train_df)} samples (subsampled). Validation on {len(val_df)} samples."
    )

    # 3. Create Datasets and Dataloaders
    train_ds = HMSDataset(train_df, config=Config, mode="train", augment=True)
    val_ds = HMSDataset(val_df, config=Config, mode="val", augment=False)
    test_ds = HMSDataset(test_df, config=Config, mode="test", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    print("Initializing model...")
    model = AsymmetricCoordinateNet(config=Config).to(device)

    # 5. Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires steps_per_epoch
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 6. Train Model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=Config,
    )

    # 7. Final Validation & Metric Calculation
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Run inference on validation set
    val_probs = inference(model, val_loader, device, Config)

    # Get targets (probabilities)
    val_targets = val_df[PROB_COLS].values.astype(np.float32)

    # Calculate Metric
    final_score = kl_divergence_loss(val_probs, val_targets)
    print(f"Final Validation Metric: {final_score}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate KL divergence per sample
    epsilon = 1e-15
    y_pred_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Term 1: P * log(P)
    term_p = np.zeros_like(val_targets)
    mask = val_targets > 0
    term_p[mask] = val_targets[mask] * np.log(val_targets[mask])

    # Term 2: P * log(Q)
    term_q = val_targets * np.log(y_pred_clipped)

    # Sum over classes
    kl_per_sample = np.sum(term_p - term_q, axis=1)

    # Add error to dataframe for correlation
    val_df_analysis = val_df.copy()
    val_df_analysis["error"] = kl_per_sample

    features_to_check = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]
    print("Correlation between Error and Input Features:")
    for feat in features_to_check:
        if feat in val_df_analysis.columns:
            corr = val_df_analysis[feat].corr(val_df_analysis["error"])
            print(f"{feat}: {corr:.4f}")

    # 9. Generate Submission
    THRESHOLD = 0.6822116374969482

    if final_score < THRESHOLD:
        print(
            f"\nScore ({final_score}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_probs = inference(model, test_loader, device, Config)

        # Create submission DataFrame
        submission = pd.DataFrame({"eeg_id": test_df["eeg_id"]})

        # Assign probabilities to the vote columns required by submission format
        # Config.CLASS_NAMES contains ['seizure_vote', 'lpd_vote', etc.]
        submission[Config.CLASS_NAMES] = test_probs

        # Ensure rows sum to 1.0 (Softmax output should be close, but normalization ensures strict compliance)
        row_sums = submission[Config.CLASS_NAMES].sum(axis=1)
        submission[Config.CLASS_NAMES] = submission[Config.CLASS_NAMES].div(
            row_sums, axis=0
        )

        # Save
        os.makedirs("./submission", exist_ok=True)
        save_path = "./submission/submission.csv"
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nScore ({final_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
