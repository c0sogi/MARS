import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.utils import AverageMeter, SAM
from library.dataset import (
    CactusDataset,
    get_transforms,
    mixup_data,
    load_and_cache_data,
)
from library.model import SelfEnsemblingRepVGG


def train_one_epoch(loader, model, criterion, optimizer, epoch, device):
    """
    Executes one epoch of training using SAM optimizer and Mixup.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        # --- SAM Step 1 ---
        # Forward pass (Main + Aux heads)
        main_out, aux_out = model(images)

        # Compute Joint Loss (Deep Supervision)
        loss_main = lam * criterion(main_out, targets_a) + (1 - lam) * criterion(
            main_out, targets_b
        )
        loss_aux = lam * criterion(aux_out, targets_a) + (1 - lam) * criterion(
            aux_out, targets_b
        )
        loss = loss_main + loss_aux

        # Backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # --- SAM Step 2 ---
        # Forward pass again on perturbed weights
        main_out_2, aux_out_2 = model(images)

        loss_main_2 = lam * criterion(main_out_2, targets_a) + (1 - lam) * criterion(
            main_out_2, targets_b
        )
        loss_aux_2 = lam * criterion(aux_out_2, targets_a) + (1 - lam) * criterion(
            aux_out_2, targets_b
        )
        loss_2 = loss_main_2 + loss_aux_2

        # Backward pass
        loss_2.backward()
        optimizer.second_step(zero_grad=True)

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(loader, model, criterion, device):
    """
    Evaluates the model using TTA and internal ensembling.
    Returns average loss and ROC AUC.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            # TTA: 4 Views (Original, HFlip, VFlip, Rot180)
            views = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]

            batch_probs = []
            for view in views:
                # Model returns (Main+Aux)/2 logits in eval mode
                logits = model(view)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs)

            # Average across TTA views
            avg_prob = torch.stack(batch_probs).mean(dim=0)

            # Compute loss on original view for tracking
            # (Using logits from original view)
            logits_orig = model(images)
            loss = criterion(logits_orig, targets)
            losses.update(loss.item(), images.size(0))

            all_preds.append(avg_prob.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc = roc_auc_score(all_targets, all_preds)
    return losses.avg, auc


def predict_test(test_loader, models, device):
    """
    Generates predictions for the test set using the ensemble of trained models and TTA.
    """
    all_preds = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # TTA Views
            views = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]

            model_ensemble_probs = []

            for model in models:
                model.eval()
                view_probs = []
                for view in views:
                    logits = model(view)
                    view_probs.append(torch.sigmoid(logits))

                # Average TTA views for this model
                model_avg = torch.stack(view_probs).mean(dim=0)
                model_ensemble_probs.append(model_avg)

            # Average across all models in the ensemble
            final_batch_preds = torch.stack(model_ensemble_probs).mean(dim=0)
            all_preds.append(final_batch_preds.cpu().numpy())

    return np.concatenate(all_preds)


def run_training():
    """
    Main execution function:
    1. Sets up environment.
    2. Loads data.
    3. Runs Stratified K-Fold Cross Validation.
    4. Trains models with SAM.
    5. Generates submission file.
    """
    Config.setup()
    device = torch.device(Config.DEVICE)

    # Load and Cache Data
    (train_imgs, train_labels), (test_imgs, test_ids) = load_and_cache_data()

    # 5-Fold Stratified Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []
    trained_models = []

    # Prepare Test Loader
    test_ds = CactusDataset(test_imgs, labels=None, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_imgs, train_labels)):
        print(f"\n--- Fold {fold + 1}/{Config.NUM_FOLDS} ---")

        # Split Data
        X_train, y_train = train_imgs[train_idx], train_labels[train_idx]
        X_val, y_val = train_imgs[val_idx], train_labels[val_idx]

        # Create Datasets and Loaders
        train_ds = CactusDataset(X_train, y_train, transform=get_transforms("train"))
        val_ds = CactusDataset(X_val, y_val, transform=get_transforms("valid"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        model = SelfEnsemblingRepVGG(num_classes=Config.NUM_CLASSES, deploy=False).to(
            device
        )

        # Initialize Optimizer (SAM wrapping AdamW)
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            rho=Config.SAM_RHO,
        )

        # Scheduler
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=Config.EPOCHS)

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None

        # Training Loop
        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, epoch, device
            )
            val_loss, val_auc = validate(val_loader, model, criterion, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Fold {fold+1} Best AUC: {best_auc}")
        fold_scores.append(best_auc)

        # Restore best model and switch to deploy mode for inference
        model.load_state_dict(best_state)
        model.switch_to_deploy()
        model.to(device)
        model.eval()
        trained_models.append(model)

        # Save checkpoint
        ckpt_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print(f"\nMean AUC across folds: {np.mean(fold_scores)}")

    # Generate Submission
    print("Generating predictions on Test set...")
    predictions = predict_test(test_loader, trained_models, device)

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": predictions.flatten()})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
