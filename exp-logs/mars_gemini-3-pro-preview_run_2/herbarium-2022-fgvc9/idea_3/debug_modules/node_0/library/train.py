import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
import numpy as np

# Import from library
from library.config import Config
from library.utils import AverageMeter, get_logger, seed_everything
from library.dataset import get_dataloaders
from library.model import HierarchicalMetricNet
from library.loss import HierarchicalMultiTaskLoss

# Initialize logger
logger = get_logger("train")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Gradient Accumulation setup
    accum_steps = Config.GRAD_ACCUM_STEPS
    optimizer.zero_grad()

    start_time = time.time()

    for batch_idx, (images, species_labels, genus_labels, family_labels) in enumerate(
        loader
    ):
        # Move data to device
        images = images.to(device, non_blocking=True)
        species_labels = species_labels.to(device, non_blocking=True)
        genus_labels = genus_labels.to(device, non_blocking=True)
        family_labels = family_labels.to(device, non_blocking=True)

        targets = (species_labels, genus_labels, family_labels)

        # Mixed Precision Forward Pass
        with autocast():
            # Pass species_labels for ArcFace margin calculation
            outputs = model(images, species_label=species_labels)
            loss, _ = criterion(outputs, targets)

            # Normalize loss for gradient accumulation
            loss = loss / accum_steps

        # Backward Pass
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update metrics (multiply back by accum_steps to log actual loss)
        loss_meter.update(loss.item() * accum_steps, images.size(0))

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                f"Epoch [{epoch}][{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})"
            )

    epoch_time = time.time() - start_time
    logger.info(
        f"Epoch {epoch} Training Complete. Avg Loss: {loss_meter.avg:.4f}. Time: {epoch_time:.2f}s"
    )

    return loss_meter.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    start_time = time.time()

    with torch.no_grad():
        for images, species_labels, genus_labels, family_labels in loader:
            images = images.to(device, non_blocking=True)
            species_labels = species_labels.to(device, non_blocking=True)
            genus_labels = genus_labels.to(device, non_blocking=True)
            family_labels = family_labels.to(device, non_blocking=True)

            targets = (species_labels, genus_labels, family_labels)

            with autocast():
                # No species_label passed -> ArcFace returns scaled cosine logits (no margin)
                outputs = model(images, species_label=None)
                loss, _ = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

            # Get predictions for species
            # outputs['species'] contains cosine similarities * scale
            preds = torch.argmax(outputs["species"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_labels.cpu().numpy())

    # Calculate Macro F1
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    val_time = time.time() - start_time
    logger.info(
        f"Validation Complete. Avg Loss: {loss_meter.avg:.4f}. Time: {val_time:.2f}s"
    )

    return loss_meter.avg, macro_f1


class Trainer:
    def __init__(self, debug=False):
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        self.debug = debug

        # Load Data
        logger.info("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader, self.meta_counts = (
            get_dataloaders(debug=debug, load_cached_data=True)
        )

        # Initialize Model
        logger.info(f"Initializing Model: {Config.MODEL_NAME}")
        self.model = HierarchicalMetricNet(
            num_species=self.meta_counts["num_species"],
            num_genera=self.meta_counts["num_genera"],
            num_families=self.meta_counts["num_families"],
        )
        self.model.to(self.device)

        # Loss, Optimizer, Scheduler
        self.criterion = HierarchicalMultiTaskLoss().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.SCHEDULER_T_MAX,
            eta_min=Config.SCHEDULER_MIN_LR,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Training State
        self.best_f1 = 0.0
        self.start_epoch = 0

    def fit(self):
        logger.info("Starting Training...")

        patience_counter = 0

        for epoch in range(self.start_epoch, Config.NUM_EPOCHS):
            logger.info(f"\n{'='*20} Epoch {epoch+1}/{Config.NUM_EPOCHS} {'='*20}")

            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.scaler,
                self.device,
                epoch + 1,
            )

            # Validate
            val_loss, val_f1 = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            logger.info(f"Epoch {epoch+1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss:   {val_loss:.6f}")
            logger.info(f"  Val F1:     {val_f1}")  # Full precision
            logger.info(f"  LR:         {current_lr:.2e}")

            # Checkpoint & Early Stopping
            if val_f1 > self.best_f1:
                logger.info(
                    f"New Best F1! ({self.best_f1} -> {val_f1}). Saving model..."
                )
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training finished. Best Val F1: {self.best_f1}")

    def predict_test_set(self):
        """
        Generates predictions for the test set using the best model.
        """
        logger.info("Loading best model for inference...")
        if os.path.exists(Config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
        else:
            logger.warning("Best model not found. Using current model state.")

        self.model.eval()

        all_ids = []
        all_preds = []

        # We need to map species_idx back to category_id
        # Load taxonomy mapping
        import pandas as pd

        cache_path = os.path.join(Config.CACHE_DIR, "taxonomy_mapping.parquet")
        if os.path.exists(cache_path):
            tax_map = pd.read_parquet(cache_path)
            # Create map: species_idx -> category_id
            idx_to_cat = dict(zip(tax_map["species_idx"], tax_map["category_id"]))
        else:
            logger.error(
                "Taxonomy mapping not found. Cannot map predictions back to category_id."
            )
            return

        logger.info("Starting inference on test set...")
        with torch.no_grad():
            for images, image_ids in self.test_loader:
                images = images.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images, species_label=None)
                    # Get species logits
                    logits = outputs["species"]
                    preds_idx = torch.argmax(logits, dim=1).cpu().numpy()

                # Map indices back to category IDs
                preds_cat = [idx_to_cat.get(idx, 0) for idx in preds_idx]

                all_ids.extend(list(image_ids))
                all_preds.extend(preds_cat)

        # Save submission
        submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

        # Ensure Id is sorted if needed, though usually order matches input
        # The sample submission has Id as int, let's ensure types match
        # image_ids from loader are strings, but competition Id is int
        try:
            submission_df["Id"] = submission_df["Id"].astype(int)
        except:
            pass  # Keep as is if conversion fails

        submission_df.sort_values("Id", inplace=True)

        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
