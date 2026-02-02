import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import library.config as config
from library.utils import save_checkpoint, load_checkpoint, quadratic_weighted_kappa
from library.dataset import create_dataloaders


class OrdinalConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny backbone with a Rank-Consistent Ordinal Regression Head.

    The model outputs K-1 logits for K classes.
    """

    def __init__(
        self,
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_ORDINAL_OUTPUTS,
        dropout_rate=config.DROPOUT_RATE,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained weights.
            num_classes (int): Number of output units (K-1 for ordinal regression).
            dropout_rate (float): Dropout rate.
        """
        super(OrdinalConvNeXt, self).__init__()

        # Load backbone with num_classes=0 to remove the default classification head.
        # global_pool='avg' ensures the output is a feature vector (Batch, Num_Features).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=dropout_rate,
        )

        # Get the number of features output by the backbone
        num_features = self.backbone.num_features

        # Custom Ordinal Head: Linear layer mapping features to K-1 outputs
        # Each output represents the logit for P(y > k)
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        # Compute logits
        logits = self.head(features)
        return logits


def create_model(model_name=config.MODEL_NAME):
    return OrdinalConvNeXt(model_name=model_name)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)  # Ordinal vectors (B, K-1)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        if config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set and computes QWK score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            targets = batch["target"].numpy()  # Integer ground truth labels

            batch_size = images.size(0)
            dataset_size += batch_size

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * batch_size

            # Decode ordinal predictions
            # Sigmoid converts logits to probabilities P(y > k)
            probs = torch.sigmoid(logits)
            # Summing probabilities gives the expected ordinal score (continuous 0 to K-1)
            scores = probs.sum(dim=1)
            # Round to nearest integer to get class label
            preds = scores.round().cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_targets.extend(targets)

    epoch_loss = running_loss / dataset_size
    qwk = quadratic_weighted_kappa(all_targets, all_preds)

    return epoch_loss, qwk


def predict_and_submit(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids = batch["id_code"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            preds = scores.round().cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_ids.extend(ids)

    df = pd.DataFrame({"id_code": all_ids, "diagnosis": all_preds})
    df.to_csv(output_path, index=False)


def main():
    """
    Main execution function for training and inference.
    """
    # Setup
    device = config.DEVICE
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config.TRAIN_META_PATH, config.VAL_META_PATH, config.TEST_META_PATH
    )

    # Model Initialization
    model = OrdinalConvNeXt().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Loss Function: Binary Cross Entropy
    # We treat each ordinal threshold as an independent binary classification task
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_score}")

        # Checkpointing & Early Stopping
        if val_score > best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score
            save_checkpoint(model, optimizer, epoch, val_score)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Score did not improve. Patience: {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Inference Phase
    print("\nStarting inference on test set...")

    # Load best model weights
    checkpoint = load_checkpoint(model)
    if checkpoint:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with score {checkpoint['score']}"
        )

    predict_and_submit(model, test_loader, device, config.SUBMISSION_PATH)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
