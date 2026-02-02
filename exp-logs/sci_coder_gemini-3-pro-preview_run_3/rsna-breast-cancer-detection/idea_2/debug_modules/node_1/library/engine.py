import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import probabilistic_f1
from library.data import get_dataloaders


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Saves the best model state.
    """

    def __init__(self, patience=3, min_delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_representation_epoch(model, dataloader, optimizer, device, criterion):
    """
    Executes one epoch of training for Stage 1 (Representation Learning).
    Updates the entire model.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def train_calibration_epoch(model, dataloader, optimizer, device, criterion):
    """
    Executes one epoch of training for Stage 2 (Calibration).
    Assumes the backbone is frozen and only the head is being updated.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets).flatten()
        all_preds = np.concatenate(all_preds).flatten()
        pf1 = probabilistic_f1(all_targets, all_preds)
    else:
        pf1 = 0.0

    return epoch_loss, pf1


def generate_submission(model, dataloader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set, aggregates them by prediction_id,
    and saves the result to a CSV file.
    """
    model.eval()
    prediction_ids = []
    probabilities = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            prediction_ids.extend(ids)
            probabilities.extend(probs)

    # Create DataFrame
    df = pd.DataFrame({"prediction_id": prediction_ids, "cancer": probabilities})

    # Aggregate by prediction_id using max pooling
    # This ensures that if any view indicates cancer, the patient score is high
    submission_df = df.groupby("prediction_id")["cancer"].max().reset_index()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run(model, device):
    """
    Orchestrates the full Two-Stage Calibration training pipeline.
    """
    criterion = nn.BCEWithLogitsLoss()

    # -------------------------------------------------------------------------
    # Stage 1: Representation Learning (Balanced Sampling)
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: Representation Learning (Balanced) ===")

    # Load Stage 1 Data (Balanced)
    loaders_s1 = get_dataloaders(stage=1, debug=Config.DEBUG)

    optimizer_s1 = torch.optim.AdamW(
        model.parameters(), lr=Config.STAGE1_LR, weight_decay=Config.STAGE1_WEIGHT_DECAY
    )
    early_stopper_s1 = EarlyStopping(patience=3, path=Config.STAGE1_CHECKPOINT)

    for epoch in range(Config.STAGE1_EPOCHS):
        train_loss = train_representation_epoch(
            model, loaders_s1["train"], optimizer_s1, device, criterion
        )
        val_loss, val_pf1 = evaluate(model, loaders_s1["val"], device, criterion)

        print(
            f"Epoch {epoch+1}/{Config.STAGE1_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        early_stopper_s1(val_loss, model)
        if early_stopper_s1.early_stop:
            print("Early stopping triggered in Stage 1.")
            break

    # Load best model from Stage 1
    print(f"Loading best Stage 1 model from {Config.STAGE1_CHECKPOINT}")
    model.load_state_dict(torch.load(Config.STAGE1_CHECKPOINT, map_location=device))

    # -------------------------------------------------------------------------
    # Stage 2: Probability Calibration (Natural Distribution)
    # -------------------------------------------------------------------------
    print("\n=== Stage 2: Probability Calibration (Natural) ===")

    # Load Stage 2 Data (Natural Distribution)
    loaders_s2 = get_dataloaders(stage=2, debug=Config.DEBUG)

    # Modify Model: Freeze Backbone, Reset Head
    model.freeze_backbone()
    model.reset_classifier()
    model.to(device)

    # Optimize only the head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_s2 = torch.optim.AdamW(trainable_params, lr=Config.STAGE2_LR)
    early_stopper_s2 = EarlyStopping(patience=3, path=Config.STAGE2_CHECKPOINT)

    for epoch in range(Config.STAGE2_EPOCHS):
        train_loss = train_calibration_epoch(
            model, loaders_s2["train"], optimizer_s2, device, criterion
        )
        val_loss, val_pf1 = evaluate(model, loaders_s2["val"], device, criterion)

        print(
            f"Epoch {epoch+1}/{Config.STAGE2_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        early_stopper_s2(val_loss, model)
        if early_stopper_s2.early_stop:
            print("Early stopping triggered in Stage 2.")
            break

    # Load best model from Stage 2
    print(f"Loading best Stage 2 model from {Config.STAGE2_CHECKPOINT}")
    model.load_state_dict(torch.load(Config.STAGE2_CHECKPOINT, map_location=device))

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    generate_submission(model, loaders_s2["test"], device)
