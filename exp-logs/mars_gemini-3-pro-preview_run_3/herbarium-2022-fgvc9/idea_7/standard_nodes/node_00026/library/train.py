import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, get_logger, calculate_macro_f1
from library.data import get_dataloaders
from library.model import HierarchicalEfficientNet

# Initialize logger
logger = get_logger("train")


class HierarchicalLoss(nn.Module):
    """
    Computes the weighted sum of CrossEntropyLoss for Species, Genus, and Family tasks.
    Applies Label Smoothing to prevent overfitting.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary of logits for 'species', 'genus', 'family'.
            targets (dict): Dictionary of target tensors for 'species', 'genus', 'family'.
        """
        loss_species = self.criterion(outputs["species"], targets["species"])
        loss_genus = self.criterion(outputs["genus"], targets["genus"])
        loss_family = self.criterion(outputs["family"], targets["family"])

        total_loss = (
            Config.WEIGHT_SPECIES * loss_species
            + Config.WEIGHT_GENUS * loss_genus
            + Config.WEIGHT_FAMILY * loss_family
        )

        return total_loss


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro F1 score for the Species task.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Get predictions for Species (Primary Task)
            preds = torch.argmax(outputs["species"], dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets["species"].cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Macro F1
    macro_f1 = calculate_macro_f1(all_targets, all_preds)

    return avg_loss, macro_f1


def fit(model, train_loader, val_loader, epochs, checkpoint_dir, stage_name):
    """
    Manages the training loop, optimizer, scheduler, and checkpointing.
    """
    device = torch.device(Config.DEVICE)
    model.to(device)

    criterion = HierarchicalLoss()

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR_MAX,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    best_f1 = -1.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    last_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    logger.info(f"Starting {stage_name} training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        logger.info(
            f"[{stage_name}] Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val F1: {val_f1}"
        )

        # Save Last Checkpoint
        torch.save(model.state_dict(), last_model_path)

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved for {stage_name} with F1: {best_f1}")

    logger.info(f"{stage_name} completed. Best Validation F1: {best_f1}")

    # Load best weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def inference(model, test_loader, device):
    """
    Generates predictions for the test set.
    Uses Horizontal Flip TTA if configured.
    """
    model.eval()
    all_ids = []
    all_preds = []

    use_tta = Config.TTA_FLIP

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            outputs = model(images)
            logits = outputs["species"]

            if use_tta:
                # Horizontal Flip TTA
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                logits_flipped = outputs_flipped["species"]

                # Average probabilities
                probs = torch.softmax(logits, dim=1)
                probs_flipped = torch.softmax(logits_flipped, dim=1)
                avg_probs = (probs + probs_flipped) / 2.0

                preds = torch.argmax(avg_probs, dim=1)
            else:
                preds = torch.argmax(logits, dim=1)

            all_ids.extend(image_ids)
            all_preds.extend(preds.cpu().numpy())

    return all_ids, all_preds


def main():
    """
    Main execution pipeline:
    1. Setup and Seeding.
    2. Stage 1 Training (224x224).
    3. Stage 2 Training (384x384).
    4. Inference and Submission Generation.
    """
    seed_everything(Config.SEED)

    # =========================================================================
    # STAGE 1: Feature Learning (224x224)
    # =========================================================================
    logger.info("Initializing Stage 1: Feature Learning (224x224)")

    train_loader_s1, val_loader_s1, _ = get_dataloaders(
        img_size=Config.STAGE1_IMG_SIZE,
        batch_size=Config.STAGE1_BATCH_SIZE,
        debug=Config.DEBUG,
    )

    model = HierarchicalEfficientNet(pretrained=True)

    model = fit(
        model=model,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        epochs=Config.STAGE1_EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR_STAGE1,
        stage_name="Stage 1",
    )

    # =========================================================================
    # STAGE 2: Fine-Grained Refinement (384x384)
    # =========================================================================
    logger.info("Initializing Stage 2: Fine-Grained Refinement (384x384)")

    # Re-initialize dataloaders with higher resolution
    train_loader_s2, val_loader_s2, test_loader = get_dataloaders(
        img_size=Config.STAGE2_IMG_SIZE,
        batch_size=Config.STAGE2_BATCH_SIZE,
        debug=Config.DEBUG,
    )

    # Continue training with the best model from Stage 1
    model = fit(
        model=model,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        epochs=Config.STAGE2_EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR_STAGE2,
        stage_name="Stage 2",
    )

    # =========================================================================
    # INFERENCE & SUBMISSION
    # =========================================================================
    logger.info("Generating predictions on Test Set...")
    device = torch.device(Config.DEVICE)

    ids, preds = inference(model, test_loader, device)

    submission_df = pd.DataFrame({"Id": ids, "Predicted": preds})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Execute the pipeline
    main()
