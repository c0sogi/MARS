import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, get_logger, accuracy
from library.dataset import CassavaDataset, get_transforms, Mixup
from library.model import CassavaModel, ModelEMA
from library.engine import train_one_epoch, valid_one_epoch


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override CFG for Fast Baseline Execution
    # We use the full dataset but limit epochs to ensure completion within ~30 mins
    CFG.debug = True
    CFG.epochs_coarse = 1
    CFG.epochs_fine = 1
    CFG.epochs = CFG.epochs_coarse + CFG.epochs_fine

    # Setup environment
    seed_everything(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    logger = get_logger(os.path.join(CFG.working_dir, "train.log"))
    logger.info(f"Starting Fast Baseline Run...")
    logger.info(
        f"Configuration: Epochs={CFG.epochs} (Coarse={CFG.epochs_coarse}, Fine={CFG.epochs_fine})"
    )
    logger.info(f"Device: {CFG.device}")

    device = torch.device(CFG.device)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading training metadata...")
    df_train_full = pd.read_csv(CFG.train_metadata_path)

    # ==========================================
    # 3. K-Fold Cross Validation
    # ==========================================
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)

    # Containers for Global Validation
    oof_preds = []
    oof_targets = []
    oof_image_ids = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_full.iloc[val_idx].reset_index(drop=True)

        # -------------------------------------------------------
        # Phase 1: Coarse Training (224px, MixUp)
        # -------------------------------------------------------
        logger.info(
            f"Phase 1: Coarse Training (Res: {CFG.img_size_coarse}, MixUp: {CFG.mixup_prob_coarse})"
        )

        train_dataset_coarse = CassavaDataset(
            df_train, transform=get_transforms("train", CFG.img_size_coarse)
        )
        # Validation during coarse phase is optional/skipped for speed, we validate at the end

        train_loader_coarse = DataLoader(
            train_dataset_coarse,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Model & Optimizer
        model = CassavaModel(pretrained=True).to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
        )

        # Initialize EMA
        model_ema = ModelEMA(model) if CFG.use_ema else None

        # MixUp Function
        mixup_fn_coarse = Mixup(
            prob=CFG.mixup_prob_coarse, switch_prob=0.5, num_classes=CFG.num_classes
        )

        for epoch in range(CFG.epochs_coarse):
            train_one_epoch(
                epoch,
                model,
                train_loader_coarse,
                optimizer,
                device,
                scheduler=scheduler,
                mixup_fn=mixup_fn_coarse,
                model_ema=model_ema,
            )
            scheduler.step()

        # -------------------------------------------------------
        # Phase 2: Fine Tuning (384px, No MixUp)
        # -------------------------------------------------------
        logger.info(
            f"Phase 2: Fine Tuning (Res: {CFG.img_size_fine}, MixUp: {CFG.mixup_prob_fine})"
        )

        # Re-initialize Datasets with Higher Resolution
        train_dataset_fine = CassavaDataset(
            df_train, transform=get_transforms("train", CFG.img_size_fine)
        )
        val_dataset_fine = CassavaDataset(
            df_val, transform=get_transforms("valid", CFG.img_size_fine)
        )

        train_loader_fine = DataLoader(
            train_dataset_fine,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader_fine = DataLoader(
            val_dataset_fine,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Disable MixUp
        mixup_fn_fine = None

        for epoch in range(CFG.epochs_coarse, CFG.epochs):
            train_one_epoch(
                epoch,
                model,
                train_loader_fine,
                optimizer,
                device,
                scheduler=scheduler,
                mixup_fn=mixup_fn_fine,
                model_ema=model_ema,
            )
            scheduler.step()

        # -------------------------------------------------------
        # Fold Validation & Save
        # -------------------------------------------------------
        # Use EMA model for inference if available
        inference_model = model_ema.ema if model_ema else model

        # Save Best Model
        save_path = os.path.join(CFG.output_dir, f"fold_{fold}_best.pth")
        torch.save(inference_model.state_dict(), save_path)

        # Generate OOF Predictions
        inference_model.eval()
        fold_preds = []
        fold_targets = []
        fold_ids = df_val["image_id"].values

        with torch.no_grad():
            for images, targets in val_loader_fine:
                images = images.to(device)
                outputs = inference_model(images)
                probs = torch.softmax(outputs, dim=1)

                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(targets.numpy())

        oof_preds.append(np.concatenate(fold_preds))
        oof_targets.append(np.concatenate(fold_targets))
        oof_image_ids.extend(fold_ids)

        # Cleanup to free memory
        del (
            model,
            model_ema,
            optimizer,
            scheduler,
            train_loader_coarse,
            train_loader_fine,
            val_loader_fine,
        )
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Global Validation & Failure Analysis
    # ==========================================
    oof_preds = np.concatenate(oof_preds)
    oof_targets = np.concatenate(oof_targets)

    # Compute Final Metric
    oof_labels = np.argmax(oof_preds, axis=1)
    final_acc = accuracy_score(oof_targets, oof_labels)

    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis: Correlation between Error and File Size
    logger.info("Performing Failure Analysis...")

    # Calculate Error (1 - Probability of True Class)
    true_class_probs = oof_preds[np.arange(len(oof_targets)), oof_targets]
    errors = 1.0 - true_class_probs

    # Retrieve File Sizes for OOF images
    # We create a mapping from image_id to file_size using the full train dataframe
    # Note: Accessing file system for 15k images is slow, so we look up paths from metadata
    img_id_to_path = dict(zip(df_train_full["image_id"], df_train_full["file_path"]))

    file_sizes = []
    for img_id in oof_image_ids:
        rel_path = img_id_to_path.get(img_id)
        full_path = os.path.join(CFG.input_dir, rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except:
            file_sizes.append(0)

    file_sizes = np.array(file_sizes)

    # Calculate Correlation
    if np.std(errors) > 0 and np.std(file_sizes) > 0:
        correlation = np.corrcoef(errors, file_sizes)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error and File Size: {correlation}")

    # ==========================================
    # 5. Submission
    # ==========================================
    if final_acc > 0.9076:
        logger.info(
            "Validation metric passed threshold (0.9076). Generating submission..."
        )

        # Load Test Data
        df_test = pd.read_csv(CFG.test_metadata_path)
        test_dataset = CassavaDataset(
            df_test,
            transform=get_transforms("test", CFG.img_size_fine),
            output_label=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Load All Fold Models
        models = []
        for fold in range(CFG.num_folds):
            model = CassavaModel(pretrained=False)
            path = os.path.join(CFG.output_dir, f"fold_{fold}_best.pth")
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)

        # Inference with TTA (Original + Horizontal Flip)
        final_preds = []

        with torch.no_grad():
            for images in tqdm(test_loader, desc="Inference"):
                images = images.to(device)
                batch_probs = torch.zeros(
                    (images.size(0), CFG.num_classes), device=device
                )

                # Standard View
                for model in models:
                    batch_probs += torch.softmax(model(images), dim=1)

                # Flipped View
                images_flip = torch.flip(images, dims=[3])
                for model in models:
                    batch_probs += torch.softmax(model(images_flip), dim=1)

                # Average (5 models * 2 views = 10)
                batch_probs /= len(models) * 2
                final_preds.append(batch_probs.cpu().numpy())

        final_preds = np.concatenate(final_preds)
        pred_labels = np.argmax(final_preds, axis=1)

        # Save Submission
        submission = pd.DataFrame(
            {"image_id": df_test["image_id"], "label": pred_labels}
        )
        sub_path = os.path.join(CFG.submission_dir, "submission.csv")
        submission.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

    else:
        logger.info(
            f"Validation metric {final_acc:.4f} did not exceed 0.9076. Submission skipped."
        )


if __name__ == "__main__":
    main()
