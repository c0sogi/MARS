import torch
import torch.nn as nn
import numpy as np
import os
from library.utils import get_score


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config):
    """
    Trains the model for one epoch using mixed precision and gradient accumulation.
    """
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    dataset_size = 0
    running_loss = 0.0

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["label"].to(device, dtype=torch.float)

        batch_size = input_ids.size(0)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            # outputs is (batch_size, 1), targets is (batch_size)
            loss = criterion(outputs.view(-1), targets)
            # Scale loss for gradient accumulation
            loss = loss / config.accumulation_steps

        # Backward Pass
        scaler.scale(loss).backward()

        # Accumulate Loss for reporting (scale back up to get actual batch loss)
        running_loss += (loss.item() * config.accumulation_steps) * batch_size
        dataset_size += batch_size

        # Gradient Accumulation Step
        if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(dataloader):
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler Step
            if scheduler is not None:
                scheduler.step()

            # Reset Gradients
            optimizer.zero_grad()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    dataset_size = 0
    running_loss = 0.0

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["label"].to(device, dtype=torch.float)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs.view(-1))

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    auc_score = get_score(all_targets, all_predictions)

    return epoch_loss, auc_score


def fit(model, train_loader, val_loader, optimizer, scheduler, device, config, fold):
    """
    Orchestrates the training loop with early stopping.
    """
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.working_dir, f"model_fold_{fold}.bin")

    print(f"Starting training for Fold {fold}...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )

        # Validate
        val_loss, val_auc = valid_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping and Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Fold {fold} training complete. Best AUC: {best_auc}")

    # Load best model weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_auc
