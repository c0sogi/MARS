import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library modules
from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.data import get_dataloaders, Mixup
from library.model import create_model, ModelEMA
from library.engine import train_one_epoch, validate
from library.inference import run_inference


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We reduce epochs and limit samples to ensure < 2 hours runtime
    Config.PHASE_1["epochs"] = 4
    Config.PHASE_2["epochs"] = 2
    MAX_TRAIN_SAMPLES = 8000

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Folds: {Config.NUM_FOLDS}")
    print(f"  Max Train Samples: {MAX_TRAIN_SAMPLES}")
    print(
        f"  Phase 1: {Config.PHASE_1['epochs']} epochs @ {Config.PHASE_1['image_size']}px"
    )
    print(
        f"  Phase 2: {Config.PHASE_2['epochs']} epochs @ {Config.PHASE_2['image_size']}px"
    )

    # =========================================================================
    # 2. Data Preparation
    # =========================================================================
    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Combine for Stratified K-Fold
    df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

    # Prepare storage for OOF predictions
    oof_preds_list = []
    oof_targets_list = []

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # =========================================================================
    # 3. Training Loop (Per Fold)
    # =========================================================================
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["label"])
    ):
        print(f"\n{'='*40}")
        print(f"Starting Fold {fold_idx + 1}/{Config.NUM_FOLDS}")
        print(f"{'='*40}")

        # Create subsets
        train_sub = df_full.iloc[train_idx].copy()
        val_sub = df_full.iloc[val_idx].copy()

        # Subsample training data for speed
        if len(train_sub) > MAX_TRAIN_SAMPLES:
            print(
                f"  Subsampling training data from {len(train_sub)} to {MAX_TRAIN_SAMPLES}..."
            )
            train_sub = train_sub.sample(n=MAX_TRAIN_SAMPLES, random_state=Config.SEED)

        # ---------------------------------------------------------------------
        # Phase 1: Coarse Feature Learning (224px, MixUp)
        # ---------------------------------------------------------------------
        print(f"\n[Fold {fold_idx}] Phase 1: Coarse Learning (224x224, MixUp On)")

        train_loader, val_loader, _ = get_dataloaders(
            train_sub, val_sub, pd.DataFrame(), Config.PHASE_1
        )

        model = create_model(pretrained=True)
        model.to(Config.DEVICE)

        model_ema = ModelEMA(model) if Config.USE_EMA else None

        optimizer = AdamW(
            model.parameters(),
            lr=Config.PHASE_1["lr"],
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.PHASE_1["epochs"], eta_min=Config.PHASE_1["min_lr"]
        )

        # Phase 1 Loss & Mixup
        loss_fn_p1 = SoftTargetCrossEntropy()
        mixup_fn = Mixup(
            mixup_alpha=Config.PHASE_1["mixup_alpha"],
            cutmix_alpha=Config.PHASE_1["cutmix_alpha"],
            prob=Config.PHASE_1["mixup_prob"],
            num_classes=Config.NUM_CLASSES,
        )

        for epoch in range(Config.PHASE_1["epochs"]):
            train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                Config.DEVICE,
                loss_fn_p1,
                mixup_fn,
                model_ema,
            )
            scheduler.step()

        # ---------------------------------------------------------------------
        # Phase Reset
        # ---------------------------------------------------------------------
        print(f"\n[Fold {fold_idx}] Phase Reset: Syncing EMA weights to Current Model")
        if model_ema:
            model_ema.reset_weights(model)

        # ---------------------------------------------------------------------
        # Phase 2: Fine-Grained Tuning (384px, No MixUp, Label Smoothing)
        # ---------------------------------------------------------------------
        print(f"\n[Fold {fold_idx}] Phase 2: Fine Tuning (384x384, MixUp Off)")

        train_loader, val_loader, _ = get_dataloaders(
            train_sub, val_sub, pd.DataFrame(), Config.PHASE_2
        )

        # Re-initialize Optimizer and Scheduler for fine-tuning
        optimizer = AdamW(
            model.parameters(),
            lr=Config.PHASE_2["lr"],
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.PHASE_2["epochs"], eta_min=Config.PHASE_2["min_lr"]
        )

        loss_fn_p2 = LabelSmoothingCrossEntropy(
            smoothing=Config.PHASE_2["label_smoothing"]
        )

        best_acc = 0.0

        for epoch in range(Config.PHASE_2["epochs"]):
            # Train (No Mixup)
            train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                Config.DEVICE,
                loss_fn_p2,
                None,
                model_ema,
            )
            scheduler.step()

            # Validate (Use EMA model if available)
            val_model = model_ema.shadow if model_ema else model
            val_loss, val_acc = validate(
                val_model, val_loader, Config.DEVICE, loss_fn_p2
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    val_model.state_dict(),
                    is_best=True,
                    checkpoint_dir=Config.CHECKPOINT_DIR,
                    fold_idx=fold_idx,
                )

        print(f"[Fold {fold_idx}] Best Validation Accuracy: {best_acc:.6f}")

        # ---------------------------------------------------------------------
        # Generate OOF Predictions
        # ---------------------------------------------------------------------
        # Reload best model
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold_idx}.pth"
        )
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(Config.DEVICE)
                outputs = model(images)
                # Store logits or probs? Let's store logits for consistency
                fold_preds.append(outputs.cpu())
                fold_targets.append(targets)

        oof_preds_list.append(torch.cat(fold_preds))
        oof_targets_list.append(torch.cat(fold_targets))

    # =========================================================================
    # 4. Global Validation & Failure Analysis
    # =========================================================================
    print(f"\n{'='*40}")
    print("Global Validation & Analysis")
    print(f"{'='*40}")

    all_preds = torch.cat(oof_preds_list)
    all_targets = torch.cat(oof_targets_list)

    # Calculate Final Metric
    pred_labels = all_preds.argmax(dim=1)
    final_acc = (pred_labels == all_targets).float().mean().item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = (pred_labels != all_targets).float().numpy()
    targets_np = all_targets.numpy()

    # Correlation between Error and Class Label
    # This helps identify if specific classes are harder
    corr = np.corrcoef(errors, targets_np)[0, 1]
    print(f"Correlation between Error Magnitude and Class Label: {corr:.6f}")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for cls_idx in range(Config.NUM_CLASSES):
        cls_mask = targets_np == cls_idx
        if cls_mask.sum() > 0:
            cls_acc = (pred_labels.numpy()[cls_mask] == cls_idx).mean()
            print(f"  Class {cls_idx}: {cls_acc:.4f}")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    THRESHOLD = 0.9076
    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric ({final_acc}) > {THRESHOLD}. Generating submission..."
        )
        # run_inference loads the best models from CHECKPOINT_DIR and generates submission.csv
        run_inference()
    else:
        print(f"\nValidation metric ({final_acc}) <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
