import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.utils import Logger, get_device, set_seed


# ==========================================
# Model Architecture
# ==========================================
class DVSEModel(nn.Module):
    """
    Dual-View Strided Ensemble (DVSE) Network.
    Uses EfficientNet-B0 as a backbone with modified input channels to handle
    2.5D stacked volumetric MRI data.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    ):
        super(DVSEModel, self).__init__()

        # Initialize EfficientNet backbone
        # timm handles weight recycling for in_chans=64 automatically
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        # Output logits (B, 1)
        return self.model(x)


# ==========================================
# Training & Validation Functions
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass (logits)
        outputs = model(inputs)

        # Squeeze to match target shape (B,)
        outputs = outputs.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for metric calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where batch might contain only one class
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs.view(-1)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


# ==========================================
# Main Pipeline
# ==========================================
def run_training(
    train_loader,
    val_loader,
    epochs=20,
    patience=5,
    learning_rate=1e-4,
    save_path="./working/best_model.pth",
):
    """
    Executes the training loop with Early Stopping.
    """
    logger = Logger()
    device = get_device()
    set_seed(42)

    logger.section("Initializing Model")
    model = DVSEModel(
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    )
    model = model.to(device)

    # Optimizer & Loss
    # Note: No weight decay as per "Idea" description
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    logger.section(f"Starting Training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        logger.log(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"  # Full precision for Val AUC
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.log(f"--> New Best AUC! Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    logger.log(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # Load best model for return
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def predict_and_submit(model, test_loader, output_path="./submission/submission.csv"):
    """
    Generates predictions using the Dual-View Ensemble strategy.
    Saves results to CSV.
    """
    logger = Logger()
    device = get_device()
    model = model.to(device)
    model.eval()

    logger.section("Starting Inference (Dual-View Ensemble)")

    ids_list = []
    probs_list = []

    with torch.no_grad():
        for img_even, img_odd, patient_ids in test_loader:
            img_even = img_even.to(device)
            img_odd = img_odd.to(device)

            # Get logits for both views
            logits_even = model(img_even).view(-1)
            logits_odd = model(img_odd).view(-1)

            # Convert to probabilities
            probs_even = torch.sigmoid(logits_even)
            probs_odd = torch.sigmoid(logits_odd)

            # Ensemble Average
            avg_probs = (probs_even + probs_odd) / 2.0

            ids_list.extend(patient_ids)
            probs_list.extend(avg_probs.cpu().numpy())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": ids_list, "MGMT_value": probs_list})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    logger.log(f"Submission saved to {output_path}")
    logger.log(f"Predicted {len(df_sub)} cases.")
