import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric, AverageMeter
from library.loss import LaplaceLogLikelihoodLoss
from library.model import OSICModel
from library.data import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for imgs, tabs, targets in loader:
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs, tabs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), imgs.size(0))

    return loss_meter.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for imgs, tabs, targets in loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            targets = targets.to(device)

            preds = model(imgs, tabs)
            loss = criterion(preds, targets)

            # Prepare for metric calculation
            fvc_pred = preds[:, 0].cpu().numpy()
            sigma_pred = preds[:, 1].cpu().numpy()
            y_true = targets.cpu().numpy().flatten()

            # Unscale predictions and targets for metric calculation
            # Cite {solution_lesson_node_00001}
            fvc_pred = fvc_pred * Config.TARGET_STD + Config.TARGET_MEAN
            y_true = y_true * Config.TARGET_STD + Config.TARGET_MEAN

            # Sigma is in the scaled space, needs to be scaled back
            # sigma_real = sigma_scaled * scale_factor
            sigma_abs = np.abs(sigma_pred) * Config.TARGET_STD

            score = laplace_log_likelihood_metric(y_true, fvc_pred, sigma_abs)

            loss_meter.update(loss.item(), imgs.size(0))
            metric_meter.update(score, imgs.size(0))

    return loss_meter.avg, metric_meter.avg


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    all_fvc = []
    all_sigma = []

    with torch.no_grad():
        for imgs, tabs, _ in test_loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            preds = model(imgs, tabs)

            fvc = preds[:, 0].cpu().numpy()
            sigma = preds[:, 1].cpu().numpy()

            all_fvc.extend(fvc)
            all_sigma.extend(sigma)

    # Access the dataframe from the dataset to preserve Patient_Week mapping
    # Note: test_loader must be shuffle=False for this to align
    sub_df = test_loader.dataset.data.copy()

    # Unscale predictions
    # Cite {solution_lesson_node_00001}
    fvc_pred = np.array(all_fvc) * Config.TARGET_STD + Config.TARGET_MEAN
    sigma_pred = np.abs(np.array(all_sigma)) * Config.TARGET_STD

    sub_df["FVC"] = fvc_pred
    # Ensure confidence is positive and clipped according to metric requirements
    sub_df["Confidence"] = np.maximum(sigma_pred, Config.SIGMA_CLIP)

    # Select required columns
    final_sub = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Orchestrates the entire training pipeline.

    Args:
        debug (bool): If True, uses a subset of data for debugging.
        epochs (int): Number of training epochs.
    """
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Starting Training | Debug: {debug} | Device: {Config.DEVICE}")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=Config.BATCH_SIZE, debug=debug
    )

    # 3. Initialize Model and Training Components
    device = torch.device(Config.DEVICE)
    model = OSICModel().to(device)

    criterion = LaplaceLogLikelihoodLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Metric: {val_metric}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Generate Submission
    print("Loading best model for submission generation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device)
