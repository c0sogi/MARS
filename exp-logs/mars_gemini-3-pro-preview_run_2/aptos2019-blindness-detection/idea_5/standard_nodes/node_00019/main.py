import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

from library.config import CFG
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_df, RetinopathyDataset, get_transforms
from library.modeling import RetinopathyModel

# --- Configuration Overrides for Fast Baseline ---
CFG.epochs = 7
CFG.swa_start_epoch_ratio = 0.6  # Start SWA after ~4 epochs
CFG.batch_size = 24
CFG.num_workers = 4


def main():
    # Setup
    seed_everything(CFG.seed)
    device = CFG.device
    print(f"Using device: {device}")

    # Load Data
    print("Loading metadata...")
    df_train_all = get_df("train")
    df_holdout = get_df("val")
    df_test = get_df("test")

    # Prepare Cross-Validation
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    # Store trained models (in memory as we have 220GB RAM)
    # Each entry: {'model': model_obj, 'type': 'cnn'|'trans'}
    ensemble_models = []

    # Training Loop
    # We iterate through folds
    for fold, (train_idx, _) in enumerate(
        skf.split(df_train_all, df_train_all["diagnosis"])
    ):
        print(f"\n================ Fold {fold}/{CFG.n_folds - 1} ================")

        # Create Fold Train Data
        df_train_fold = df_train_all.iloc[train_idx].reset_index(drop=True)

        # --- Stream 1: CNN (EfficientNet-B5) ---
        print(f"Training Stream 1: CNN (EfficientNet-B5) | Fold {fold}")

        # Dataset & Loader
        ds_cnn = RetinopathyDataset(
            df_train_fold, transform=get_transforms("train", CFG.img_size_cnn)
        )
        dl_cnn = DataLoader(
            ds_cnn,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Model, Optimizer, Scheduler
        model_cnn = RetinopathyModel(CFG.model_cnn_name, pretrained=True).to(device)
        opt_cnn = optim.AdamW(
            model_cnn.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        sched_cnn = optim.lr_scheduler.CosineAnnealingLR(
            opt_cnn, T_max=CFG.epochs, eta_min=CFG.min_lr
        )

        # SWA Setup
        swa_model_cnn = AveragedModel(model_cnn)
        swa_sched_cnn = SWALR(opt_cnn, swa_lr=CFG.swa_lr)
        swa_start = int(CFG.epochs * CFG.swa_start_epoch_ratio)

        scaler = GradScaler()
        criterion = nn.MSELoss()

        for epoch in range(CFG.epochs):
            model_cnn.train()
            losses = []
            for imgs, labels in dl_cnn:
                imgs, labels = imgs.to(device), labels.to(device)

                opt_cnn.zero_grad()
                with autocast():
                    preds = model_cnn(imgs)
                    loss = criterion(preds.view(-1), labels.view(-1))

                scaler.scale(loss).backward()
                if CFG.max_grad_norm > 0:
                    scaler.unscale_(opt_cnn)
                    torch.nn.utils.clip_grad_norm_(
                        model_cnn.parameters(), CFG.max_grad_norm
                    )
                scaler.step(opt_cnn)
                scaler.update()
                losses.append(loss.item())

            # Scheduler Step
            if epoch >= swa_start:
                swa_model_cnn.update_parameters(model_cnn)
                swa_sched_cnn.step()
            else:
                sched_cnn.step()

        # Update BN for SWA Model
        print("  Updating SWA Batch Norm statistics...")
        torch.optim.swa_utils.update_bn(dl_cnn, swa_model_cnn, device=device)
        ensemble_models.append({"model": swa_model_cnn, "type": "cnn"})

        # Cleanup
        del model_cnn, opt_cnn, sched_cnn, dl_cnn, ds_cnn, scaler
        torch.cuda.empty_cache()

        # --- Stream 2: Transformer (Swin V2) ---
        print(f"Training Stream 2: Transformer (Swin V2) | Fold {fold}")

        ds_trans = RetinopathyDataset(
            df_train_fold, transform=get_transforms("train", CFG.img_size_trans)
        )
        dl_trans = DataLoader(
            ds_trans,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        model_trans = RetinopathyModel(CFG.model_trans_name, pretrained=True).to(device)
        opt_trans = optim.AdamW(
            model_trans.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        sched_trans = optim.lr_scheduler.CosineAnnealingLR(
            opt_trans, T_max=CFG.epochs, eta_min=CFG.min_lr
        )

        swa_model_trans = AveragedModel(model_trans)
        swa_sched_trans = SWALR(opt_trans, swa_lr=CFG.swa_lr)

        scaler = GradScaler()

        for epoch in range(CFG.epochs):
            model_trans.train()
            losses = []
            for imgs, labels in dl_trans:
                imgs, labels = imgs.to(device), labels.to(device)

                opt_trans.zero_grad()
                with autocast():
                    preds = model_trans(imgs)
                    loss = criterion(preds.view(-1), labels.view(-1))

                scaler.scale(loss).backward()
                if CFG.max_grad_norm > 0:
                    scaler.unscale_(opt_trans)
                    torch.nn.utils.clip_grad_norm_(
                        model_trans.parameters(), CFG.max_grad_norm
                    )
                scaler.step(opt_trans)
                scaler.update()
                losses.append(loss.item())

            if epoch >= swa_start:
                swa_model_trans.update_parameters(model_trans)
                swa_sched_trans.step()
            else:
                sched_trans.step()

        print("  Updating SWA Batch Norm statistics...")
        torch.optim.swa_utils.update_bn(dl_trans, swa_model_trans, device=device)
        ensemble_models.append({"model": swa_model_trans, "type": "trans"})

        del model_trans, opt_trans, sched_trans, dl_trans, ds_trans, scaler
        torch.cuda.empty_cache()

    # --- Inference Function ---
    def predict(models, df, use_tta=True):
        # Prepare Loaders
        ds_cnn = RetinopathyDataset(
            df, transform=get_transforms("val", CFG.img_size_cnn)
        )
        dl_cnn = DataLoader(
            ds_cnn,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )

        ds_trans = RetinopathyDataset(
            df, transform=get_transforms("val", CFG.img_size_trans)
        )
        dl_trans = DataLoader(
            ds_trans,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )

        n_samples = len(df)
        agg_preds = np.zeros(n_samples)

        # Group models
        cnn_models = [m["model"] for m in models if m["type"] == "cnn"]
        trans_models = [m["model"] for m in models if m["type"] == "trans"]

        # Inference CNN
        if cnn_models:
            for m in cnn_models:
                m.eval()
            ptr = 0
            with torch.no_grad():
                for batch in dl_cnn:
                    # Handle dataset returning (img, label) or (img)
                    if isinstance(batch, (tuple, list)):
                        imgs = batch[0]
                    else:
                        imgs = batch
                    imgs = imgs.to(device)
                    bs = imgs.size(0)

                    batch_p = np.zeros(bs)
                    for m in cnn_models:
                        batch_p += m(imgs).view(-1).cpu().numpy()
                        if use_tta:
                            batch_p += m(torch.flip(imgs, [3])).view(-1).cpu().numpy()

                    agg_preds[ptr : ptr + bs] += batch_p
                    ptr += bs

        # Inference Trans
        if trans_models:
            for m in trans_models:
                m.eval()
            ptr = 0
            with torch.no_grad():
                for batch in dl_trans:
                    if isinstance(batch, (tuple, list)):
                        imgs = batch[0]
                    else:
                        imgs = batch
                    imgs = imgs.to(device)
                    bs = imgs.size(0)

                    batch_p = np.zeros(bs)
                    for m in trans_models:
                        batch_p += m(imgs).view(-1).cpu().numpy()
                        if use_tta:
                            batch_p += m(torch.flip(imgs, [3])).view(-1).cpu().numpy()

                    agg_preds[ptr : ptr + bs] += batch_p
                    ptr += bs

        # Normalize
        n_votes = len(models) * (2 if use_tta else 1)
        return agg_preds / n_votes

    # --- Validation ---
    print("\n=== Performing Validation ===")
    val_preds_raw = predict(ensemble_models, df_holdout, use_tta=CFG.use_tta)
    val_targets = df_holdout["diagnosis"].values

    # Metric
    score = quadratic_weighted_kappa(val_targets, val_preds_raw)
    print(f"Final Validation Metric: {score}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")
    residuals = np.abs(val_targets - val_preds_raw)

    # Correlation with Diagnosis (Target)
    if len(np.unique(val_targets)) > 1:
        corr_diag, _ = pearsonr(residuals, val_targets)
        print(f"Correlation (Error vs Diagnosis): {corr_diag:.4f}")
    else:
        print("Correlation (Error vs Diagnosis): N/A (Single class)")

    # Correlation with Prediction Magnitude
    corr_pred, _ = pearsonr(residuals, val_preds_raw)
    print(f"Correlation (Error vs Prediction): {corr_pred:.4f}")

    # --- Submission ---
    threshold = 0.9207435978935975
    if score > threshold:
        print("\nScore meets threshold. Generating submission...")
        test_preds_raw = predict(ensemble_models, df_test, use_tta=CFG.use_tta)

        # Round to nearest int and clip
        test_preds_int = np.round(test_preds_raw).astype(int)
        test_preds_int = np.clip(test_preds_int, 0, 4)

        submission = pd.DataFrame(
            {"id_code": df_test["id_code"], "diagnosis": test_preds_int}
        )

        os.makedirs(os.path.dirname(CFG.submission_path), exist_ok=True)
        submission.to_csv(CFG.submission_path, index=False)
        print(f"Submission saved to {CFG.submission_path}")
    else:
        print(
            f"\nScore {score:.4f} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
