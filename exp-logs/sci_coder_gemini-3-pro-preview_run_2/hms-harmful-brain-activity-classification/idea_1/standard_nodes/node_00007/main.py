import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import albumentations as A
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config
from library.dataset import EEGDataset
from library.model import SpecEfficientNet
from library.engine import Trainer


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_transforms():
    """
    Returns albumentations transforms.
    The EEGProcessor outputs [0, 1], so we normalize to ImageNet stats.
    Includes CoarseDropout for regularization (Cite Lesson 00006).
    """
    return A.Compose(
        [
            A.CoarseDropout(
                max_holes=8,
                max_height=Config.IMG_SIZE[0] // 8,
                max_width=Config.IMG_SIZE[1] // 8,
                min_holes=1,
                fill_value=0,
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,  # Input is already [0, 1]
                p=1.0,
            ),
        ]
    )


def calculate_kl_divergence(preds, targets):
    """
    Calculates KL Divergence between predictions and targets.
    Metric = sum(target * log(target / prediction))
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-6
    preds = np.clip(preds, epsilon, 1 - epsilon)

    # Calculate KL: y_true * (log(y_true) - log(y_pred))
    # Handle y_true = 0 case where limit is 0
    # We use np.where to handle the 0 * log(0) case safely

    # Term 1: y_true * log(y_true)
    # If y_true is 0, term is 0. Else y_true * log(y_true)
    term1 = np.zeros_like(targets)
    mask = targets > 0
    term1[mask] = targets[mask] * np.log(targets[mask])

    # Term 2: y_true * log(y_pred)
    term2 = targets * np.log(preds)

    # KL = Sum(Term1 - Term2)
    kl_per_sample = np.sum(term1 - term2, axis=1)
    return kl_per_sample


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Use full training data for maximum performance
    # N_TRAIN_SAMPLES set to >81k to include all data
    N_TRAIN_SAMPLES = 100000
    if len(train_df) > N_TRAIN_SAMPLES:
        print(
            f"Subsampling training set from {len(train_df)} to {N_TRAIN_SAMPLES} samples."
        )
        train_df = train_df.sample(
            n=N_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Datasets & Loaders
    transforms = get_transforms()

    train_dataset = EEGDataset(train_df, mode="train", transform=transforms)
    val_dataset = EEGDataset(val_df, mode="val", transform=transforms)
    test_dataset = EEGDataset(test_df, mode="test", transform=transforms)

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
    print("Initializing model...")
    model = SpecEfficientNet(config=Config, pretrained=True)
    model.to(device)

    # 5. Training Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR for fast convergence
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Cite solution_lesson_node_00003: Enable MixUp (alpha=0.4) to combat overfitting
    trainer = Trainer(model, optimizer, device, scheduler, mixup_alpha=0.4)

    # 6. Train
    print("Starting training...")
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=save_path,
    )

    # Load best model for validation/inference
    model.load_state_dict(torch.load(save_path, weights_only=True))

    # 7. Validation & Metric
    print("Performing final validation...")
    val_probs = trainer.predict(val_loader)
    val_targets = val_df[Config.TARGET_COLS].values

    # Calculate KL Divergence
    kl_errors = calculate_kl_divergence(val_probs, val_targets)
    mean_kl = np.mean(kl_errors)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {mean_kl}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Add errors to dataframe
    val_analysis = val_df.copy()
    val_analysis["kl_error"] = kl_errors

    # Check correlations
    features_to_check = ["total_votes", "eeg_label_offset_seconds"]
    features_to_check = [f for f in features_to_check if f in val_analysis.columns]

    print("Correlation between Error (KL) and Metadata Features:")
    for feat in features_to_check:
        corr = val_analysis["kl_error"].corr(val_analysis[feat])
        print(f"  {feat}: {corr:.4f}")

    # 9. Inference & Submission
    if mean_kl < 0.8172267878622345:
        print("\nGenerating submission...")
        test_probs = trainer.predict(test_loader)

        submission_df = pd.DataFrame(
            {
                "eeg_id": test_df["eeg_id"],
                "seizure_vote": test_probs[:, 0],
                "lpd_vote": test_probs[:, 1],
                "gpd_vote": test_probs[:, 2],
                "lrda_vote": test_probs[:, 3],
                "grda_vote": test_probs[:, 4],
                "other_vote": test_probs[:, 5],
            }
        )

        # Verify sum to 1
        submission_df.iloc[:, 1:] = submission_df.iloc[:, 1:].div(
            submission_df.iloc[:, 1:].sum(axis=1), axis=0
        )

        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nSkipping submission: Validation KL {mean_kl:.5f} did not improve baseline (0.81723)."
        )


if __name__ == "__main__":
    main()
