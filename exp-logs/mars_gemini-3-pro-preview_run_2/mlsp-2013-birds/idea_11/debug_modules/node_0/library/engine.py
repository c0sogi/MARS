import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_auc


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss as the weighted sum of losses for the two targets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, data_loader, device, criterion, scheduler=None):
    """
    Executes one epoch of training.
    """
    model.train()
    total_loss = 0.0

    use_mixup = Config.USE_MIXUP
    mixup_alpha = Config.MIXUP_ALPHA

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, mixup_alpha, device
            )
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            # Step scheduler if it's per-iteration (though Config uses 'constant')
            # For standard epoch-based schedulers, this might be called outside.
            # We include it here for flexibility.
            pass

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    print(f"Training Loss: {avg_loss}")

    return avg_loss


def validate(model, data_loader, device, criterion):
    """
    Evaluates the model on the validation set.
    Prints Loss and AUC with full precision.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    auc = calculate_auc(all_targets, all_preds)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation AUC: {auc}")

    return avg_loss, auc


def inference(model, data_loader, device):
    """
    Generates predictions for the test set.
    Returns recording IDs and probability arrays.
    """
    model.eval()
    all_probs = []
    all_rec_ids = []

    with torch.no_grad():
        for inputs, rec_ids in data_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_rec_ids.append(rec_ids.numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
        all_rec_ids = np.concatenate(all_rec_ids, axis=0)
    else:
        all_probs = np.array([])
        all_rec_ids = np.array([])

    return all_rec_ids, all_probs


def save_predictions(rec_ids, probs, output_path):
    """
    Formats predictions into the submission CSV format and saves to disk.
    Format: Id,Probability where Id = rec_id * 100 + species_id
    """
    submission_rows = []
    num_classes = probs.shape[1]

    for i, rec_id in enumerate(rec_ids):
        for species_id in range(num_classes):
            # Construct the unique Id for the submission
            row_id = int(rec_id * 100 + species_id)
            prob = probs[i, species_id]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)

    # Sort by Id for consistency
    df_sub = df_sub.sort_values("Id")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
