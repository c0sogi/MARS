import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import defaultdict

from library.config import Config
from library.utils import get_logger, set_seed, get_device, ProbabilisticF1
from library.data import get_dataloaders
from library.model import MTSIN

logger = get_logger(name="train")


class MultiTaskLoss(nn.Module):
    """
    Computes the weighted sum of losses for the primary task (Cancer)
    and auxiliary tasks (BIRADS, Density).
    """

    def __init__(self, device):
        super(MultiTaskLoss, self).__init__()

        # Primary Task: Cancer (Binary Classification)
        # Using pos_weight to handle extreme class imbalance
        self.cancer_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device)
        )

        # Aux Task: BIRADS (Multi-class)
        # ignore_index=-1 handles missing labels
        self.birads_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Aux Task: Density (Multi-class)
        self.density_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.lambda_birads = Config.LAMBDA_BIRADS
        self.lambda_density = Config.LAMBDA_DENSITY

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary of logits from the model.
            targets (dict): Dictionary of target tensors.
        """
        # 1. Cancer Loss
        # Target shape needs to match logit shape (B, 1)
        cancer_loss = self.cancer_criterion(
            outputs["cancer"], targets["cancer"].view(-1, 1)
        )

        # 2. BIRADS Loss
        birads_loss = self.birads_criterion(outputs["birads"], targets["birads"])

        # 3. Density Loss
        density_loss = self.density_criterion(outputs["density"], targets["density"])

        # Total Loss
        total_loss = (
            cancer_loss
            + (self.lambda_birads * birads_loss)
            + (self.lambda_density * density_loss)
        )

        return total_loss, cancer_loss.item(), birads_loss.item(), density_loss.item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_cancer_loss = 0.0

    # Iterate over batches (tqdm removed as per instructions)
    for batch_idx, batch in enumerate(loader):
        # Move inputs to device
        images = batch["image"].to(device)
        meta = batch["meta"].to(device)

        targets = {
            "cancer": batch["target_cancer"].to(device),
            "birads": batch["target_birads"].to(device),
            "density": batch["target_density"].to(device),
        }

        optimizer.zero_grad()

        # Forward pass with Mixed Precision
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(images, meta)
            loss, c_loss, _, _ = criterion(outputs, targets)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()

        running_loss += loss.item()
        running_cancer_loss += c_loss

    avg_loss = running_loss / len(loader)
    avg_cancer_loss = running_cancer_loss / len(loader)

    return avg_loss, avg_cancer_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    pf1_metric = ProbabilisticF1()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)

            targets = {
                "cancer": batch["target_cancer"].to(device),
                "birads": batch["target_birads"].to(device),
                "density": batch["target_density"].to(device),
            }

            outputs = model(images, meta)

            # Compute Loss
            loss, _, _, _ = criterion(outputs, targets)
            running_loss += loss.item()

            # Compute Metrics (only on Cancer head)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs["cancer"])
            pf1_metric.update(probs, targets["cancer"])

    avg_loss = running_loss / len(loader)
    pf1_score = pf1_metric.compute()

    return avg_loss, pf1_score


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions for the test set and creates the submission file.
    Aggregates predictions by prediction_id using Max Pooling.
    """
    logger.info("Starting inference on test set...")
    model.eval()

    # Dictionary to store probabilities: {prediction_id: [prob1, prob2, ...]}
    predictions_map = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            prediction_ids = batch["prediction_id"]  # List of strings

            outputs = model(images, meta)
            probs = torch.sigmoid(outputs["cancer"]).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                predictions_map[pid].append(prob)

    # Aggregate predictions (Max Pooling)
    final_preds = []
    for pid, probs_list in predictions_map.items():
        # Max probability across all views for this prediction_id
        max_prob = np.max(probs_list)
        final_preds.append({"prediction_id": pid, "cancer": max_prob})

    # Create DataFrame
    submission_df = pd.DataFrame(final_preds)

    # Ensure column order
    submission_df = submission_df[["prediction_id", "cancer"]]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Head:\n{submission_df.head()}")


def run_training(debug=False):
    set_seed(Config.SEED)
    device = get_device()

    logger.info(f"Running training. Debug mode: {debug}")
    logger.info(f"Device: {device}")

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # 2. Model
    model = MTSIN()
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 4. Loss
    criterion = MultiTaskLoss(device)

    # 5. Training Loop
    best_pf1 = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_cancer_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler=scaler
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} (Cancer: {train_cancer_loss:.6f}) | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val pF1: {val_pf1:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best pF1! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # 6. Inference
    logger.info("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning("Best model file not found. Using current model state.")

    predict_and_submit(model, test_loader, device)


if __name__ == "__main__":
    # This block is here for local testing if run directly,
    # but the prompt asks to only implement the module functions.
    # The logic is encapsulated in run_training.
    pass
