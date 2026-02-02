import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library import utils, data, model


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: [B, 1]

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to logits to get probabilities for AUC
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle case where only one class is present in batch (rare but possible in small debug subsets)
        auc_score = 0.5

    return total_loss, auc_score


def get_model_optimizer_scheduler(device):
    """
    Instantiates the model, optimizer, and scheduler based on Config.
    """
    # Initialize Model
    net = model.CustomWideResNeSt()
    net = net.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_ETA_MIN
    )

    return net, optimizer, scheduler


def train_seed(seed, device):
    """
    Runs the training pipeline for a single seed.
    Saves the best model to disk.
    """
    print(f"\n--- Starting Training for Seed {seed} ---")
    utils.set_seed(seed)

    # Data Loaders
    train_loader, _ = data.get_dataloader("train", shuffle=True)
    val_loader, _ = data.get_dataloader("val", shuffle=False)

    # Model setup
    net, optimizer, scheduler = get_model_optimizer_scheduler(device)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(net, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(net.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_auc}")

    print(f"Finished Seed {seed}. Best Val AUC: {best_auc}")
    return best_model_path


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (Original + HFlip + VFlip).
    Returns IDs and averaged probabilities.
    """
    model.eval()
    all_probs = []

    # We need the IDs corresponding to the test set order
    # The dataloader returns (images, labels), but labels are placeholders.
    # The ids are returned by get_dataloader separately, but we need to ensure alignment.
    # Since shuffle=False for test, the order is preserved.

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # 1. Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            inputs_h = torch.flip(inputs, dims=[3])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2 is height)
            inputs_v = torch.flip(inputs, dims=[2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            prob_avg = (prob_orig + prob_h + prob_v) / 3.0

            all_probs.append(prob_avg.cpu().numpy())

    return np.concatenate(all_probs).flatten()


def run_ensemble():
    """
    Main driver function.
    Trains models for all seeds, generates predictions, averages them, and saves submission.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    ensemble_probs = []
    test_ids = None

    # Get Test Loader once (order is fixed)
    test_loader, ids = data.get_dataloader("test", shuffle=False)
    test_ids = ids

    # Iterate over seeds
    for seed in Config.SEEDS:
        # 1. Train
        best_model_path = train_seed(seed, device)

        # 2. Load Best Model for Inference
        net = model.CustomWideResNeSt()
        net.load_state_dict(torch.load(best_model_path, map_location=device))
        net = net.to(device)

        # 3. Predict with TTA
        print(f"Generating predictions for Seed {seed}...")
        probs = predict_with_tta(net, test_loader, device)
        ensemble_probs.append(probs)

        # Clean up to save memory
        del net
        torch.cuda.empty_cache()

    # 4. Average Predictions
    print("Averaging ensemble predictions...")
    ensemble_probs = np.array(ensemble_probs)  # Shape: [Num_Seeds, Num_Samples]
    avg_probs = np.mean(ensemble_probs, axis=0)

    # 5. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    utils.save_submission(test_ids, avg_probs)
    print("Done.")
