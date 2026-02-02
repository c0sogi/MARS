import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library import config
from library import utils
from library import dataset
from library import model


def train_one_epoch(model_instance, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model_instance.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model_instance(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = utils.calculate_roc_auc(all_targets, all_preds)
    except ValueError:
        # Handle edge case with single class in batch
        auc = 0.5

    return epoch_loss, auc


def evaluate(model_instance, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model_instance.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model_instance(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = utils.calculate_roc_auc(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def predict_with_tta(model_instance, loader, device):
    """
    Generates predictions using Test Time Augmentation (Original + HFlip + VFlip).
    """
    model_instance.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # 1. Original
            out1 = model_instance(images)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip (dim 3 is width)
            img_h = torch.flip(images, [3])
            out2 = model_instance(img_h)
            prob2 = torch.sigmoid(out2)

            # 3. Vertical Flip (dim 2 is height)
            img_v = torch.flip(images, [2])
            out3 = model_instance(img_v)
            prob3 = torch.sigmoid(out3)

            # Average probabilities
            avg_prob = (prob1 + prob2 + prob3) / 3.0

            all_preds.append(avg_prob.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds), all_ids


def run():
    """
    Main execution function:
    1. Sets up data loaders.
    2. Trains models across multiple seeds (Homogeneous Seed Averaging).
    3. Performs TTA inference.
    4. Saves submission.
    """
    # 1. Setup
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # We use load_cached_data=True to leverage the caching mechanism in dataset.py
    train_dataset = dataset.CactusDataset(
        config.TRAIN_METADATA_PATH,
        phase="train",
        transform=dataset.get_transforms("train"),
        load_cached_data=True,
    )
    val_dataset = dataset.CactusDataset(
        config.VAL_METADATA_PATH,
        phase="val",
        transform=dataset.get_transforms("val"),
        load_cached_data=True,
    )
    test_dataset = dataset.CactusDataset(
        config.TEST_METADATA_PATH,
        phase="test",
        transform=dataset.get_transforms("test"),
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Accumulator for averaged predictions
    final_test_preds = np.zeros((len(test_dataset), 1))
    test_ids_list = None

    # 3. Training Loop over Seeds
    for seed in config.SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        utils.set_seed(seed)

        # Initialize Model
        net = model.WideSEResNet(
            num_classes=config.NUM_CLASSES,
            stages=config.MODEL_PARAMS["stages"],
            se_reduction=config.MODEL_PARAMS["se_reduction"],
            use_gap=config.MODEL_PARAMS["use_gap"],
            dropout_rate=config.MODEL_PARAMS["dropout_rate"],
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )

        best_auc = 0.0
        patience_counter = 0
        best_checkpoint_filename = f"model_seed_{seed}.pth"

        for epoch in range(config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                net, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = evaluate(net, val_loader, criterion, device)

            scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss} AUC: {train_auc} | Val Loss: {val_loss} AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0

                # Save checkpoint (copies to model_best.pth)
                utils.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "best_auc": best_auc,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    is_best=True,
                    checkpoint_dir=config.CHECKPOINT_DIR,
                    filename=best_checkpoint_filename,
                )
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 4. Inference for this seed
        print(f"Loading best model for seed {seed} and predicting...")
        best_model_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pth")

        # Reload best weights
        utils.load_checkpoint(best_model_path, net, device=device)

        # Predict with TTA
        preds, ids = predict_with_tta(net, test_loader, device)

        final_test_preds += preds
        if test_ids_list is None:
            test_ids_list = ids

    # 5. Average Predictions and Save Submission
    final_test_preds /= len(config.SEEDS)

    submission_df = pd.DataFrame(
        {"id": test_ids_list, "has_cactus": final_test_preds.flatten()}
    )

    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
