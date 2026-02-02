import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from timm.data import Mixup
from timm.utils import ModelEmaV2
import warnings

# Import provided library modules
from library import config, utils, data, model, engine

# Suppress warnings
warnings.filterwarnings("ignore")


def run():
    # 1. Configuration
    cfg = config.CFG
    # Override epochs to ensure quick baseline execution within 2 hours
    cfg.epochs = 4
    cfg.output_dir = "./working/idea_6"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set seed
    utils.seed_everything(cfg.seed)

    # Logger
    logger = utils.get_logger(os.path.join(cfg.output_dir, "run.log"))
    logger.info(f"Starting training with {cfg.epochs} epochs per fold.")

    device = cfg.device

    # 2. Prepare Data (Folds)
    # This generates/loads folds.parquet in cfg.output_dir
    df_folds = data.prepare_folds(load_cached_data=True)

    # 3. Training Loop (5 Folds)
    for fold in range(cfg.n_fold):
        logger.info(f"========== Fold {fold} ==========")

        # Get DataLoaders
        train_loader, valid_loader = data.get_loaders(fold, load_cached_data=True)

        # Initialize Model
        net = model.CassavaClassifier(model_name=cfg.model_name, pretrained=True)
        net.to(device)

        # Initialize EMA
        model_ema = ModelEmaV2(net, decay=cfg.ema_decay, device=device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )

        # Mixup / Cutmix
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.0,
            num_classes=cfg.num_classes,
        )

        # AMP Scaler
        scaler = torch.amp.GradScaler("cuda")

        best_acc = 0

        for epoch in range(cfg.epochs):
            # Train
            engine.train_one_epoch(
                epoch,
                net,
                optimizer,
                train_loader,
                device,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                scaler=scaler,
            )

            # Validate (using EMA model)
            val_loss, val_acc = engine.valid_one_epoch(
                model_ema.module, valid_loader, device
            )

            scheduler.step()

            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    model_ema.module.state_dict(),
                    os.path.join(cfg.output_dir, f"fold_{fold}_best.pth"),
                )

        logger.info(f"Fold {fold} Best Accuracy: {best_acc:.4f}")

        # Cleanup to free memory
        del net, model_ema, optimizer, scheduler, scaler, train_loader, valid_loader
        torch.cuda.empty_cache()

    # 4. Validation Analysis
    logger.info("========== Validation Analysis ==========")

    # Load hold-out validation metadata
    val_meta_path = os.path.join(cfg.metadata_dir, "val.csv")
    df_val_holdout = pd.read_csv(val_meta_path)

    # Map image_id to fold to perform OOF inference
    img_to_fold = dict(zip(df_folds.image_id, df_folds.fold))

    # Group validation images by their assigned fold
    fold_to_imgs = {}
    for img_id in df_val_holdout["image_id"]:
        f = img_to_fold.get(img_id)
        if f is not None:
            if f not in fold_to_imgs:
                fold_to_imgs[f] = []
            fold_to_imgs[f].append(img_id)

    val_preds = []
    val_targets = []
    val_ids = []

    # Iterate through folds to predict on the corresponding validation subset
    for fold in range(cfg.n_fold):
        if fold not in fold_to_imgs:
            continue

        # Subset of val.csv that belongs to this fold
        img_ids = fold_to_imgs[fold]
        subset_df = df_val_holdout[
            df_val_holdout["image_id"].isin(img_ids)
        ].reset_index(drop=True)

        if len(subset_df) == 0:
            continue

        # Create Dataset & Loader for this subset
        ds = data.CassavaDataset(subset_df, transform=data.get_transforms(data="valid"))
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        # Load Best Model for this fold
        net = model.CassavaClassifier(model_name=cfg.model_name, pretrained=False)
        ckpt_path = os.path.join(cfg.output_dir, f"fold_{fold}_best.pth")
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.to(device)
        net.eval()

        # Predict
        preds = engine.predict(net, loader, device)

        val_preds.append(preds)
        val_targets.append(torch.tensor(subset_df["label"].values))
        val_ids.extend(subset_df["image_id"].values)

        del net, loader, ds
        torch.cuda.empty_cache()

    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)

    # Calculate Final Metric
    acc1 = utils.accuracy(val_preds, val_targets, topk=(1,))[0]
    final_metric = acc1 / 100.0
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis: Correlation between error and file size
    # Error is defined as (1.0 - probability of correct class)
    # val_preds are probabilities from engine.predict
    correct_probs = val_preds[range(len(val_targets)), val_targets]
    errors = 1.0 - correct_probs.cpu().numpy()

    file_sizes = []
    for img_id in val_ids:
        # Validation images are located in train_images directory
        path = os.path.join(cfg.input_dir, "train_images", img_id)
        if os.path.exists(path):
            file_sizes.append(os.path.getsize(path))
        else:
            file_sizes.append(0)

    if len(file_sizes) > 0:
        corr = np.corrcoef(errors, file_sizes)[0, 1]
        print(f"Correlation between Error and File Size: {corr:.4f}")

    # 5. Submission
    if final_metric > 0.8995994659546062:
        logger.info("Generating Submission...")

        test_loader = data.get_test_loader()
        test_preds = []

        # Ensemble predictions from all 5 folds
        for fold in range(cfg.n_fold):
            net = model.CassavaClassifier(model_name=cfg.model_name, pretrained=False)
            ckpt_path = os.path.join(cfg.output_dir, f"fold_{fold}_best.pth")
            net.load_state_dict(torch.load(ckpt_path, map_location=device))
            net.to(device)
            net.eval()

            # Predict
            fold_pred = engine.predict(net, test_loader, device)
            test_preds.append(fold_pred)

            del net
            torch.cuda.empty_cache()

        # Average predictions
        avg_preds = torch.stack(test_preds).mean(dim=0)
        pred_labels = avg_preds.argmax(dim=1).cpu().numpy()

        # Create submission dataframe
        sub_df = pd.read_csv(os.path.join(cfg.metadata_dir, "test.csv"))
        sub_df["label"] = pred_labels

        # Save
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_df[["image_id", "label"]].to_csv(
            os.path.join(sub_dir, "submission.csv"), index=False
        )
        logger.info("Submission saved.")
    else:
        logger.info("Validation metric too low. Skipping submission.")


if __name__ == "__main__":
    run()
