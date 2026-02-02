import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from timm.utils import ModelEmaV2
import numpy as np

from library.config import Config
from library.utils import get_logger, calculate_metrics, set_seed
from library.dataset import get_dataloaders, get_mixup_fn
from library.model import create_model
from library.loss import ClassBalancedFocalLoss


class Trainer:
    """
    Trainer class for Animal Classification.
    Handles training loop, validation, EMA, and checkpointing.
    """

    def __init__(self, debug=False):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, runs a shorter version of the training for debugging.
        """
        self.debug = debug
        self.device = Config.get_device()
        self.logger = get_logger(log_file=os.path.join(Config.WORKING_DIR, "train.log"))

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Data Loaders
        self.logger.info("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # 2. Model
        self.logger.info(f"Creating model: {Config.MODEL_NAME}")
        self.model = create_model().to(self.device)

        # 3. EMA (Exponential Moving Average)
        self.ema_model = None
        if Config.USE_EMA:
            self.logger.info("Initializing Model EMA...")
            # ModelEmaV2 is more efficient and handles buffers correctly
            self.ema_model = ModelEmaV2(self.model, decay=Config.EMA_DECAY)

        # 4. Loss Function
        self.logger.info(f"Initializing Loss: {Config.LOSS_TYPE}")
        self.criterion = ClassBalancedFocalLoss().to(self.device)

        # 5. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 6. Mixup / CutMix
        self.mixup_fn = get_mixup_fn()

        # 7. Scaler for AMP
        self.scaler = GradScaler(enabled=Config.USE_AMP)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()

        running_loss = 0.0
        num_batches = len(self.train_loader)

        start_time = time.time()

        for i, (images, targets) in enumerate(self.train_loader):
            if self.debug and i > 10:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply Mixup/CutMix
            if self.mixup_fn is not None:
                images, targets = self.mixup_fn(images, targets)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=Config.USE_AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward Pass
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA
            if self.ema_model is not None and (i % Config.EMA_UPDATE_EVERY == 0):
                self.ema_model.update(self.model)

            running_loss += loss.item()

        avg_loss = running_loss / num_batches if not self.debug else running_loss / 10
        duration = time.time() - start_time

        self.logger.info(
            f"Epoch [{epoch}] Train Loss: {avg_loss:.6f} | Time: {duration:.2f}s"
        )
        return avg_loss

    def validate(self, model_to_eval, loader):
        """
        Runs validation on the given model and loader.

        Args:
            model_to_eval: The model instance (nn.Module) to evaluate.
            loader: DataLoader to use.

        Returns:
            tuple: (avg_loss, macro_f1)
        """
        model_to_eval.eval()

        all_preds = []
        all_targets = []
        running_loss = 0.0
        num_batches = len(loader)

        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                if self.debug and i > 10:
                    break

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Standard Forward Pass (No Mixup during Val)
                with autocast(enabled=Config.USE_AMP):
                    outputs = model_to_eval(images)
                    # Loss calculation requires targets to be handled carefully if they were mixed up
                    # But here in validation, targets are standard indices.
                    # Our ClassBalancedFocalLoss handles indices correctly.
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = running_loss / num_batches if not self.debug else running_loss / 10

        # Calculate Metric
        macro_f1 = calculate_metrics(np.array(all_targets), np.array(all_preds))

        return avg_loss, macro_f1

    def fit(self, epochs=Config.EPOCHS, patience=5):
        """
        Main training loop with Early Stopping.

        Args:
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        self.logger.info(f"Starting training for {epochs} epochs...")

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=Config.MIN_LR
        )
        # Warmup scheduler could be added, but CosineAnnealing is usually sufficient
        # combined with a lower starting LR or flat start.
        # For simplicity adhering to the prompt's request for flexibility, we stick to standard Cosine.

        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            # If using EMA, we primarily want to track EMA performance as that's our inference model
            # But checking base model performance is also good for debugging.
            # We will use EMA for the 'best model' decision if enabled.

            eval_model = self.ema_model.module if self.ema_model else self.model
            val_loss, val_f1 = self.validate(eval_model, self.val_loader)

            # Step Scheduler
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch [{epoch}] Val Loss: {val_loss:.6f} | Val F1: {val_f1} | LR: {current_lr:.2e}"
            )

            # Checkpointing & Early Stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self.logger.info(f"New best F1: {best_f1}. Saving model...")

                # Save Best Model
                # If EMA is used, we save the EMA weights as the best model for inference
                state_dict = eval_model.state_dict()
                torch.save(state_dict, Config.BEST_MODEL_PATH)

                # Also save the EMA wrapper state to resume training if needed (optional)
                if self.ema_model:
                    torch.save(self.ema_model.state_dict(), Config.EMA_MODEL_PATH)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training completed. Best Val F1: {best_f1}")

    def predict_test_set(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves predictions to submission.csv.
        """
        self.logger.info("Generating predictions for test set...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            self.logger.warning("Best model not found. Using current model weights.")
            model_to_use = self.ema_model.module if self.ema_model else self.model
        else:
            self.logger.info(f"Loading best model from {Config.BEST_MODEL_PATH}")
            # We load into the base model structure
            checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            model_to_use = self.model

        model_to_use.eval()

        predictions = []
        ids = []

        with torch.no_grad():
            for i, (images, _) in enumerate(self.test_loader):
                if self.debug and i > 5:
                    break

                images = images.to(self.device, non_blocking=True)

                # Inference
                with autocast(enabled=Config.USE_AMP):
                    outputs = model_to_use(images)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                # Get Ids for this batch
                # The dataset doesn't return IDs in __getitem__, so we rely on the order
                # The DataLoader is sequential (shuffle=False), so we can map indices back to the dataframe
                start_idx = i * Config.BATCH_SIZE
                end_idx = start_idx + len(preds)
                batch_ids = self.test_loader.dataset.df.iloc[start_idx:end_idx][
                    "Id"
                ].values

                predictions.extend(preds)
                ids.extend(batch_ids)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"Id": ids, "Predicted": predictions})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
