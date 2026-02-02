import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from library.config import Config
from library.utils import get_logger
from library.model_backbone import AWP

logger = get_logger("Engine")


def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, epoch, awp=None, scaler=None
):
    """
    Trains the model for one epoch using Mixed Precision and AWP.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    # Initialize Scaler for AMP if not provided
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    criterion = nn.MSELoss()

    pbar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}"
    )

    for step, batch in pbar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # 1. Standard Forward Pass with Mixed Precision
        with torch.amp.autocast("cuda", dtype=Config.dtype):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1), labels.view(-1))

            # Scale loss for gradient accumulation
            loss = loss / Config.accum_iter

        # 2. Standard Backward Pass
        scaler.scale(loss).backward()

        # 3. Adversarial Weight Perturbation (AWP)
        if awp is not None and epoch >= Config.awp_start_epoch:
            # Perturb weights based on gradients from the standard pass
            awp.attack_step()

            # Forward pass with perturbed weights
            with torch.amp.autocast("cuda", dtype=Config.dtype):
                logits_adv = model(input_ids, attention_mask)
                loss_adv = criterion(logits_adv.view(-1), labels.view(-1))
                loss_adv = loss_adv / Config.accum_iter

            # Backward pass with perturbed weights (accumulate gradients)
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # 4. Optimizer Step (with Gradient Accumulation)
        if (step + 1) % Config.accum_iter == 0 or (step + 1) == len(dataloader):
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad_norm)

            # Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update metrics
        # We multiply by accum_iter to get back the actual loss for reporting
        running_loss += (loss.item() * Config.accum_iter) * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        pbar.set_postfix(loss=epoch_loss, lr=optimizer.param_groups[0]["lr"])

    return running_loss / dataset_size


def valid_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    preds = []
    targets = []

    criterion = nn.MSELoss()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid")

    with torch.no_grad():
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            # Forward Pass
            with torch.amp.autocast("cuda", dtype=Config.dtype):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1), labels.view(-1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions
            preds.append(logits.view(-1).float().cpu().numpy())
            targets.append(labels.view(-1).float().cpu().numpy())

            epoch_loss = running_loss / dataset_size
            pbar.set_postfix(loss=epoch_loss)

    preds = np.concatenate(preds)

    return running_loss / dataset_size, preds


def extract_embeddings(model, dataloader, device):
    """
    Extracts embeddings from the model for Stacking.
    Returns a numpy array of shape (n_samples, hidden_size).
    """
    model.eval()
    embeddings = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Extract Embeddings")

    with torch.no_grad():
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward Pass with return_embedding=True
            with torch.amp.autocast("cuda", dtype=Config.dtype):
                # The model's forward method handles pooling and returns the feature vector
                features = model(input_ids, attention_mask, return_embedding=True)

            embeddings.append(features.float().cpu().numpy())

    return np.concatenate(embeddings, axis=0)
