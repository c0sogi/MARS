import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, StepLR

# Import provided library functions
from library.utils import seed_everything, calculate_f1
from library.dataset import get_dataloaders
from library.model import HerbariumEfficientNet
from library.trainer import Trainer
from library.inference import predict_and_submit

# Configuration
SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGE1_EPOCHS = 1
STAGE2_EPOCHS = 1
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
THRESHOLD = 0.4137111055501958


def analyze_failures(val_df, preds, train_df):
    """
    Analyzes failure modes by correlating errors with class frequency.
    """
    print("\n==== Failure Analysis ====")

    # Calculate class frequencies from training data
    class_counts = train_df["category_id"].value_counts().to_dict()

    # Prepare validation analysis dataframe
    # val_df has columns: image_id, file_path, category_id
    analysis_df = val_df.copy()

    # Map class frequency
    analysis_df["class_freq"] = analysis_df["category_id"].map(class_counts)
    analysis_df["class_freq"] = analysis_df["class_freq"].fillna(0)

    # Add predictions
    # Note: preds is a list of class INDICES. We need to map them back to category_ids if we want to compare directly,
    # but for error calculation, we just need to know if pred == target.
    # The val_loader returns labels as indices (0..N-1), so we need the mapped targets from the loader or
    # map the dataframe category_id to indices.
    # However, since we have the ground truth category_id in val_df, let's just assume
    # we need to compare mapped predictions.
    # Actually, simpler: The validation loop below returns targets and preds as INDICES.
    # We can just add 'is_error' column.

    analysis_df["is_error"] = (
        analysis_df["target_idx"] != analysis_df["pred_idx"]
    ).astype(int)

    # Correlation
    # We correlate error (1 for error, 0 for correct) with class frequency.
    # A negative correlation is expected (higher freq -> lower error).
    corr_freq = analysis_df["is_error"].corr(analysis_df["class_freq"])

    print(f"Correlation between Error and Class Frequency: {corr_freq:.6f}")

    # Additional stats
    avg_error_rare = analysis_df[analysis_df["class_freq"] < 20]["is_error"].mean()
    avg_error_common = analysis_df[analysis_df["class_freq"] >= 20]["is_error"].mean()
    print(f"Error Rate on Rare Classes (<20 samples): {avg_error_rare:.4f}")
    print(f"Error Rate on Common Classes (>=20 samples): {avg_error_common:.4f}")


def main():
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # Ensure working directories exist
    os.makedirs("./working/stage1", exist_ok=True)
    os.makedirs("./working/stage2", exist_ok=True)

    # ==========================================
    # STAGE 1: Representation Learning
    # ==========================================
    print("\n[Stage 1] Starting Representation Learning (Square-Root Sampling)...")

    # Load Data with Square-Root Sampling
    train_loader_s1, val_loader, classes = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampling_mode="sqrt",
        load_cached_data=True,
        debug=False,
    )

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    # Initialize Model
    model = HerbariumEfficientNet(num_classes=num_classes)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=STAGE1_LR)
    steps_per_epoch = len(train_loader_s1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=STAGE1_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=STAGE1_EPOCHS,
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Trainer
    trainer_s1 = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        classes=classes,
        save_dir="./working/stage1",
    )

    # Train
    trainer_s1.fit(
        train_loader_s1,
        val_loader,
        epochs=STAGE1_EPOCHS,
        checkpoint_name="model_stage1.pth",
    )

    # ==========================================
    # STAGE 2: Classifier Re-balancing
    # ==========================================
    print("\n[Stage 2] Starting Classifier Re-balancing (Class-Balanced Sampling)...")

    # Load Data with Class-Balanced Sampling
    # Note: We reuse val_loader as it doesn't change
    train_loader_s2, _, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampling_mode="balanced",
        load_cached_data=True,
        debug=False,
    )

    # Load Best Stage 1 Model (Trainer.fit already loads the best state into self.model)
    # But to be explicit and safe, we use the model instance currently in memory which is the best from Stage 1.

    # Freeze Backbone
    model.freeze_backbone()
    print("Backbone frozen. Fine-tuning classifier head.")

    # Re-initialize Optimizer for the head only
    # Filter parameters to only those requiring grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_s2 = optim.AdamW(trainable_params, lr=STAGE2_LR)

    # Scheduler: Constant or simple decay. Since it's 1 epoch, essentially constant.
    scheduler_s2 = StepLR(optimizer_s2, step_size=1, gamma=0.1)

    # Trainer for Stage 2
    trainer_s2 = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer_s2,
        scheduler=scheduler_s2,
        device=DEVICE,
        classes=classes,
        save_dir="./working/stage2",
    )

    # Train
    trainer_s2.fit(
        train_loader_s2,
        val_loader,
        epochs=STAGE2_EPOCHS,
        checkpoint_name="model_stage2.pth",
    )

    # ==========================================
    # Final Validation & Analysis
    # ==========================================
    print("\n[Evaluation] Running Final Validation...")

    model.eval()
    all_targets = []
    all_preds = []

    # We need to manually run inference to get arrays for analysis
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_targets.extend(targets.numpy())
            all_preds.extend(preds)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    final_f1 = calculate_f1(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    # Load metadata for analysis
    val_df = pd.read_csv("./metadata/val.csv")
    train_df = pd.read_csv("./metadata/train.csv")

    # Ensure alignment (val_loader is sequential shuffle=False)
    if len(val_df) == len(all_targets):
        val_df["target_idx"] = all_targets
        val_df["pred_idx"] = all_preds
        analyze_failures(val_df, all_preds, train_df)
    else:
        print(
            "Warning: Validation dataframe length mismatch with predictions. Skipping detailed failure analysis."
        )

    # ==========================================
    # Submission
    # ==========================================
    if final_f1 > THRESHOLD:
        print(f"\nMetric {final_f1} > Threshold {THRESHOLD}. Generating submission...")

        # Path to the best model from Stage 2
        checkpoint_path = "./working/stage2/model_stage2.pth"
        output_file = "./submission/submission.csv"

        predict_and_submit(
            checkpoint_path=checkpoint_path,
            output_file=output_file,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=DEVICE,
            debug=False,
        )
    else:
        print(f"\nMetric {final_f1} <= Threshold {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
