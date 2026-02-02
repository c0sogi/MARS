import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import Logger, get_device, set_seed
from library.data import get_dataloaders
from library.model import DVSEModel, train_one_epoch, predict_and_submit


def validate_ensemble(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using Multi-View Ensemble.
    Since the validation set is constructed as [Patient1_ViewA, Patient1_ViewB, Patient2_ViewA, ...],
    we collect predictions and average every pair to compute the patient-level AUC.
    """
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

    # Convert to numpy for reshaping
    probs_np = np.array(all_probs)
    targets_np = np.array(all_targets)

    # Reshape to (N_patients, 2) to group View A and View B
    # The validation loader is not shuffled, preserving the A, B order.
    if len(probs_np) % 2 == 0:
        probs_reshaped = probs_np.reshape(-1, 2)
        targets_reshaped = targets_np.reshape(-1, 2)

        # Average probabilities across views
        avg_probs = probs_reshaped.mean(axis=1)
        # Targets are identical for both views, take the first col
        patient_targets = targets_reshaped[:, 0]
    else:
        # Fallback in case of unexpected dataset size (unlikely with current data.py)
        avg_probs = probs_np
        patient_targets = targets_np

    try:
        epoch_auc = roc_auc_score(patient_targets, avg_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run(
    epochs=20,
    patience=5,
    learning_rate=1e-4,
    batch_size=16,
    debug_limit=None,
    save_path="./working/best_model.pth",
    submission_path="./submission/submission.csv",
):
    """
    Main execution function for training and submission.
    """
    # 1. Setup
    set_seed(42)
    device = get_device()
    logger = Logger()

    # 2. Data Loading
    # Caching is handled internally by get_dataloaders/process_dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, debug_limit=debug_limit
    )

    # 3. Model Initialization
    logger.section("Initializing DVSE Model")
    model = DVSEModel(
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    )
    model = model.to(device)

    # 4. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    logger.section(f"Starting Training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate (Ensemble)
        val_loss, val_auc = validate_ensemble(model, val_loader, criterion, device)

        logger.log(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.log(f"--> New Best AUC! Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log(f"Early stopping triggered after {patience} epochs.")
                break

    logger.log(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # 6. Inference & Submission
    if os.path.exists(save_path):
        logger.log("Loading best model for inference...")
        model.load_state_dict(torch.load(save_path, map_location=device))

        predict_and_submit(model, test_loader, output_path=submission_path)
    else:
        logger.log("No model file found. Skipping inference.")
