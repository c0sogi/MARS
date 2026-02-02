import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

# Import from library
from library.config import (
    SEED,
    DEVICE,
    WORKING_DIR,
    CHECKPOINTS_DIR,
    SUBMISSION_PATH,
    CACHE_DIR,
    N_FOLDS,
    EPOCHS_STANDARD,
    EPOCHS_SWA,
    SWA_LR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    TTA_VIEWS,
    TRAIN_METADATA_CSV,
    VAL_METADATA_CSV,
    INPUT_DIR,
)
from library.utils import seed_everything, calculate_roc_auc
from library.data_loader import (
    get_fold_dataloaders,
    get_test_dataloader,
    load_dataset_arrays,
)
from library.model import MultiScaleRepVGG, reparameterize_model
from library.engine import train_one_epoch, evaluate, SWAHandler


def run_tta_inference(model, loader, device):
    """
    Runs inference with Test Time Augmentation (4 views).
    Returns averaged probabilities.
    """
    model.eval()
    all_preds = []

    # TTA Flips: Original, H, V, HV (Rot180)

    with torch.no_grad():
        for batch in loader:
            # Handle tuple (img, label) or just img
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            batch_preds = []

            # 1. Original
            out = model(images)
            # Aggregate heads
            probs = torch.stack([torch.sigmoid(o).view(-1) for o in out]).mean(dim=0)
            batch_preds.append(probs)

            # 2. Horizontal Flip
            images_h = torch.flip(images, [3])
            out_h = model(images_h)
            probs_h = torch.stack([torch.sigmoid(o).view(-1) for o in out_h]).mean(
                dim=0
            )
            batch_preds.append(probs_h)

            # 3. Vertical Flip
            images_v = torch.flip(images, [2])
            out_v = model(images_v)
            probs_v = torch.stack([torch.sigmoid(o).view(-1) for o in out_v]).mean(
                dim=0
            )
            batch_preds.append(probs_v)

            # 4. Rot 180 (H + V)
            images_hv = torch.flip(images, [2, 3])
            out_hv = model(images_hv)
            probs_hv = torch.stack([torch.sigmoid(o).view(-1) for o in out_hv]).mean(
                dim=0
            )
            batch_preds.append(probs_hv)

            # Average across views
            avg_batch_preds = torch.stack(batch_preds).mean(dim=0)
            all_preds.extend(avg_batch_preds.cpu().numpy())

    return np.array(all_preds)


def main():
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Starting 5-Fold Training on {DEVICE}...")

    # Load full data arrays once to get IDs for OOF tracking
    # We need to replicate the split logic to map predictions to IDs
    full_imgs, full_labels, full_ids = load_dataset_arrays(
        load_cached_data=True, is_test=False
    )

    # Containers for OOF
    oof_ids = []
    oof_preds = []
    oof_targets = []

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(full_imgs, full_labels))

    # 2. Training Loop
    for fold in range(N_FOLDS):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

        # DataLoaders
        train_loader, val_loader = get_fold_dataloaders(fold, load_cached_data=True)

        # Model
        model = MultiScaleRepVGG(deploy=False).to(DEVICE)

        # Optimizer & Scheduler (Phase 1)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_STANDARD
        )

        # --- Phase 1: Standard Training ---
        print("Phase 1: Standard Training")
        for epoch in range(EPOCHS_STANDARD):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE, epoch
            )
            scheduler.step()

        # --- Phase 2: SWA Training ---
        print("Phase 2: SWA Training")
        swa_handler = SWAHandler(model, optimizer, swa_lr=SWA_LR)

        for epoch in range(EPOCHS_SWA):
            # Train one epoch (optimizer updates weights)
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                DEVICE,
                EPOCHS_STANDARD + epoch,
            )
            # Update SWA model (averages weights)
            swa_handler.update()

        # --- Finalize Fold ---
        print("Finalizing SWA Model...")
        # Update BN statistics for SWA model
        swa_handler.update_bn(train_loader, DEVICE)

        # Get the averaged model
        final_model = swa_handler.get_averaged_model()

        # Reparameterize for inference (Fuse blocks)
        # SWA model is wrapped in AveragedModel, access module
        final_model_core = final_model.module
        deploy_model = reparameterize_model(final_model_core)

        # Save Checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}_model.pth")
        torch.save(deploy_model.state_dict(), ckpt_path)
        print(f"Saved fold model to {ckpt_path}")

        # --- Validation (OOF) ---
        print("Generating OOF predictions...")
        val_preds = run_tta_inference(deploy_model, val_loader, DEVICE)

        # Get Validation IDs and Targets
        _, val_idx = splits[fold]
        fold_val_ids = full_ids[val_idx]
        fold_val_targets = full_labels[val_idx]

        oof_ids.extend(fold_val_ids)
        oof_preds.extend(val_preds)
        oof_targets.extend(fold_val_targets)

        fold_auc = calculate_roc_auc(fold_val_targets, val_preds)
        print(f"Fold {fold} ROC AUC: {fold_auc:.10f}")

    # 3. Overall Evaluation
    oof_ids = np.array(oof_ids)
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    final_auc = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_auc:.16f}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Load metadata to get features
    df_train = pd.read_csv(TRAIN_METADATA_CSV)
    df_val = pd.read_csv(VAL_METADATA_CSV)
    df_meta = pd.concat([df_train, df_val], ignore_index=True)

    # Create OOF DataFrame
    df_oof = pd.DataFrame({"id": oof_ids, "pred": oof_preds, "target": oof_targets})

    # Calculate Error
    df_oof["error"] = (df_oof["pred"] - df_oof["target"]).abs()

    # Merge with metadata
    df_merged = pd.merge(df_oof, df_meta, on="id", how="left")

    # Calculate file sizes
    file_sizes = []
    for _, row in df_merged.iterrows():
        fpath = os.path.join(INPUT_DIR, row["file_path"])
        if os.path.exists(fpath):
            file_sizes.append(os.path.getsize(fpath))
        else:
            file_sizes.append(0)
    df_merged["file_size"] = file_sizes

    # Correlation
    if "file_size" in df_merged.columns:
        corr = df_merged["error"].corr(df_merged["file_size"])
        print(f"Correlation between Error Magnitude and File Size: {corr:.4f}")

    # 5. Submission
    print("\n=== Generating Submission ===")
    test_loader, test_ids = get_test_dataloader(load_cached_data=True)

    # Accumulate predictions from all folds
    final_test_preds = np.zeros(len(test_ids))

    for fold in range(N_FOLDS):
        print(f"Predicting with Fold {fold} model...")
        # Load model
        model = MultiScaleRepVGG(deploy=True).to(DEVICE)  # Initialize in deploy mode
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"fold_{fold}_model.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        # Predict
        fold_preds = run_tta_inference(model, test_loader, DEVICE)
        final_test_preds += fold_preds

    # Average
    final_test_preds /= N_FOLDS

    # Save
    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_test_preds})
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
