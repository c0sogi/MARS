import os
import torch
import torch.nn as nn
import numpy as np
import copy
from library.config import Config
from library.utils import average_weights, calculate_metric
from library.model import DogClassifier
from library.data import get_dataloaders


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Fold Training - Epoch {epoch} Loss: {epoch_loss}")
    return epoch_loss


def valid_one_epoch(model, loader, device):
    """
    Performs validation and calculates Log Loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    # Use CrossEntropyLoss for running loss tracking, but calculate_metric for final score
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            # Apply softmax to get probabilities for Log Loss metric
            probs = torch.softmax(logits, dim=1)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate official metric (Log Loss)
    metric_score = calculate_metric(all_labels, all_preds)

    return metric_score, all_preds


def create_greedy_soup(model, loader, checkpoint_paths, device):
    """
    Constructs a Greedy Model Soup from a list of checkpoints.

    1. Evaluate all models.
    2. Sort by performance.
    3. Iteratively average weights and keep if performance improves.
    """
    print(
        f"Starting Greedy Soup construction with {len(checkpoint_paths)} candidates..."
    )

    # 1. Evaluate all individual checkpoints
    candidates = []
    for path in checkpoint_paths:
        print(f"Evaluating candidate: {path}")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        loss, _ = valid_one_epoch(model, loader, device)
        candidates.append({"path": path, "loss": loss, "state_dict": state_dict})
        print(f"Candidate Loss: {loss}")

    # 2. Sort by Loss (Ascending)
    candidates.sort(key=lambda x: x["loss"])

    if not candidates:
        print("No candidates available for soup.")
        return model.state_dict()

    # 3. Greedy Selection
    # Start with the best model
    soup_state = candidates[0]["state_dict"]
    best_loss = candidates[0]["loss"]
    print(f"Initial Soup Base: {candidates[0]['path']} with Loss: {best_loss}")

    ingredients = [candidates[0]["path"]]

    for i in range(1, len(candidates)):
        candidate = candidates[i]
        print(
            f"Attempting to add {candidate['path']} (Individual Loss: {candidate['loss']})..."
        )

        # Create potential new soup
        new_soup_state = average_weights(soup_state, candidate["state_dict"])

        # Load and Evaluate
        model.load_state_dict(new_soup_state)
        current_loss, _ = valid_one_epoch(model, loader, device)

        if current_loss < best_loss:
            print(f"Soup Improved! Loss dropped from {best_loss} to {current_loss}")
            best_loss = current_loss
            soup_state = new_soup_state
            ingredients.append(candidate["path"])
        else:
            print(f"Soup did not improve (Loss: {current_loss}). Rejecting candidate.")

    print(f"Final Soup constructed from {len(ingredients)} models.")
    print(f"Final Soup Log Loss: {best_loss}")

    return soup_state


def train_fold(fold_idx):
    """
    Orchestrates the training process for a single fold, including:
    - Warm-up phase
    - Fine-tuning phase
    - Checkpointing
    - Greedy Soup construction
    """
    print(f"--- Starting Training for Fold {fold_idx} ---")

    device = torch.device(Config.device)

    # 1. Data
    train_loader, val_loader, _ = get_dataloaders(fold_idx)

    # 2. Model
    model = DogClassifier(pretrained=True).to(device)

    # 3. Phase 1: Warm-up (Head only)
    print("Phase 1: Warm-up (Training Head Only)")
    model.freeze_backbone()

    # Optimizer for head only
    optimizer_head = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=Config.weight_decay,
    )

    # Train 1 epoch for warm-up
    train_one_epoch(model, train_loader, optimizer_head, device, epoch="Warmup")
    val_loss, _ = valid_one_epoch(model, val_loader, device)
    print(f"Warmup Validation Log Loss: {val_loss}")

    # 4. Phase 2: Fine-tuning (Full Model)
    print("Phase 2: Fine-tuning (Full Model)")
    model.unfreeze_backbone()

    # Optimizer for full model with conservative LR
    optimizer = torch.optim.AdamW(
        model.get_optimizer_params(Config.learning_rate),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_max, eta_min=Config.min_lr
    )

    # Directory for checkpoints
    checkpoint_dir = os.path.join(Config.working_dir, f"fold_{fold_idx}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    soup_candidates = []

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, _ = valid_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch} - Train Loss: {train_loss} - Val Log Loss: {val_loss}")

        scheduler.step()

        # Save checkpoints for Soup (last N epochs)
        if epoch >= Config.soup_start_epoch:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            soup_candidates.append(ckpt_path)
            print(f"Saved checkpoint for soup: {ckpt_path}")

    # 5. Create Greedy Soup
    if soup_candidates:
        print("Constructing Greedy Model Soup...")
        best_soup_state = create_greedy_soup(model, val_loader, soup_candidates, device)

        # Save Final Model for this Fold
        final_model_path = os.path.join(
            Config.working_dir, f"best_model_fold_{fold_idx}.pth"
        )
        torch.save(best_soup_state, final_model_path)
        print(f"Fold {fold_idx} Complete. Best Soup Model saved to {final_model_path}")
    else:
        # Fallback if no soup candidates (e.g., short run)
        final_model_path = os.path.join(
            Config.working_dir, f"best_model_fold_{fold_idx}.pth"
        )
        torch.save(model.state_dict(), final_model_path)
        print(f"Fold {fold_idx} Complete. Last Model saved to {final_model_path}")
