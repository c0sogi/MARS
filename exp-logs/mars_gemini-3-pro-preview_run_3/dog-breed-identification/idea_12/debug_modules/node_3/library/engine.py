import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """
    Trains the model for one epoch using Cross Entropy Loss.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (or string for warmup).
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    logger.info(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average Log Loss on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    logger.info(f"Validation Loss: {avg_loss}")

    return avg_loss


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Averages predictions from the original image and a horizontally flipped version.

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        np.ndarray: Array of predicted probabilities (N_samples, N_classes).
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Forward pass - Original
            outputs_orig = model(images)
            probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Forward pass - Horizontal Flip
            # Flip along width dimension (dim 3 for NCHW)
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # 3. Average Probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            preds_list.append(probs_avg.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


def greedy_model_soup(model, dataloader, checkpoints, device):
    """
    Constructs a Greedy Model Soup from a list of checkpoints.
    Iteratively adds models to the soup if they improve validation loss.

    Args:
        model: The model architecture instance.
        dataloader: Validation dataloader.
        checkpoints: List of dicts {'state_dict': dict, 'loss': float}.
        device: torch device.

    Returns:
        dict: The state_dict of the constructed soup model.
    """
    logger.info(f"Starting Greedy Model Soup with {len(checkpoints)} candidates...")

    # 1. Sort checkpoints by validation loss (ascending)
    checkpoints.sort(key=lambda x: x["loss"])

    # 2. Initialize soup with the single best model
    best_model_dict = checkpoints[0]["state_dict"]
    best_loss = checkpoints[0]["loss"]

    # List to keep track of state dicts currently in the soup
    soup_ingredients = [best_model_dict]

    logger.info(f"Baseline (Best Single Model) Loss: {best_loss}")

    # 3. Iteratively try adding other models
    for i in range(1, len(checkpoints)):
        candidate_dict = checkpoints[i]["state_dict"]

        # Create potential new soup ingredients (Current Soup + Candidate)
        current_ingredients = soup_ingredients + [candidate_dict]

        # Calculate average weights for the candidate soup
        avg_state_dict = copy.deepcopy(current_ingredients[0])
        for key in avg_state_dict:
            # Stack tensors from all ingredients and take the mean
            tensors = [d[key] for d in current_ingredients]
            avg_state_dict[key] = torch.stack(tensors).mean(dim=0)

        # Load candidate soup into model and evaluate
        model.load_state_dict(avg_state_dict)
        current_loss = evaluate(model, dataloader, device)

        logger.info(
            f"Candidate {i} (Individual Loss: {checkpoints[i]['loss']}) -> Soup Loss: {current_loss}"
        )

        # Greedy Selection: Keep if loss improves
        if current_loss < best_loss:
            logger.info(f"  -> Improvement! Added candidate {i} to soup.")
            best_loss = current_loss
            soup_ingredients.append(candidate_dict)
        else:
            logger.info(f"  -> No improvement. Discarding candidate {i}.")

    # 4. Construct Final Soup
    final_state_dict = copy.deepcopy(soup_ingredients[0])
    for key in final_state_dict:
        tensors = [d[key] for d in soup_ingredients]
        final_state_dict[key] = torch.stack(tensors).mean(dim=0)

    logger.info(f"Final Soup Loss: {best_loss}")
    return final_state_dict


def train_fold(model, train_loader, val_loader, device, fold_idx):
    """
    Orchestrates the full training pipeline for a single fold:
    Warmup -> Fine-tuning -> Greedy Model Soup.

    Args:
        model: The PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        device: 'cuda' or 'cpu'.
        fold_idx: Index of the current fold.

    Returns:
        model: The model loaded with the best soup weights.
    """
    logger.info(f"=== Starting Fold {fold_idx} ===")

    # --- Phase 1: Warmup ---
    # Freeze backbone, train head with higher LR to align weights
    logger.info("Phase 1: Warmup (Frozen Backbone)")
    model.freeze_backbone()

    # Use standard LR for head initialization
    warmup_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    for epoch in range(Config.WARMUP_EPOCHS):
        train_one_epoch(
            model, train_loader, warmup_optimizer, device, f"Warmup-{epoch}"
        )
        evaluate(model, val_loader, device)

    # --- Phase 2: Fine-tuning ---
    # Unfreeze backbone, train with conservative LR and Cosine Annealing
    logger.info("Phase 2: Fine-tuning (Unfrozen Backbone)")
    model.unfreeze_backbone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Buffer to store checkpoints for Model Soup
    candidate_checkpoints = []

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scheduler
        )
        val_loss = evaluate(model, val_loader, device)

        # Save checkpoints from the last N epochs for the soup
        if epoch >= (Config.EPOCHS - Config.SOUP_CANDIDATES):
            # Deepcopy and move to CPU to save GPU memory
            state_dict = copy.deepcopy(model.state_dict())
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

            candidate_checkpoints.append(
                {"state_dict": state_dict, "loss": val_loss, "epoch": epoch}
            )

    # --- Phase 3: Greedy Model Soup ---
    logger.info("Phase 3: Greedy Model Soup Construction")

    # Construct soup using the validation set
    best_soup_state = greedy_model_soup(
        model, val_loader, candidate_checkpoints, device
    )

    # Load the best soup weights into the model
    model.load_state_dict(best_soup_state)

    return model


def save_submission(predictions, test_ids, breed_columns, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Probability matrix.
        test_ids (list): List of test image IDs.
        breed_columns (list): List of breed names (column headers).
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame(predictions, columns=breed_columns)
    df.insert(0, "id", test_ids)
    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
