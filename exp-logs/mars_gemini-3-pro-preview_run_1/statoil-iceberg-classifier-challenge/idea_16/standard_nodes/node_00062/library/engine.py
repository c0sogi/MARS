import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from library.config import Config
from library.utils import calculate_log_loss, save_checkpoint


class IcebergTrainer:
    """
    Trainer class for the Iceberg vs Ship classification task.
    Handles training, validation, SWA, and prediction.
    """

    def __init__(self, model, device=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (str, optional): Device to run on ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device if device else Config.DEVICE
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, loader, optimizer):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch in loader:
            # Unpack batch: (images, angles, labels)
            images, angles, labels = batch

            images = images.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Shape (B, 1)

            optimizer.zero_grad()
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    def evaluate(self, loader, model=None):
        """
        Evaluates the model on the validation set.

        Args:
            loader: DataLoader for validation.
            model: Optional model to evaluate (e.g., SWA model).

        Returns:
            avg_loss, log_loss, predictions
        """
        eval_model = model if model else self.model
        eval_model.eval()

        running_loss = 0.0
        preds = []
        targets = []
        dataset_size = 0

        with torch.no_grad():
            for batch in loader:
                images, angles, labels = batch

                images = images.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = eval_model(images, angles)
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                probs = torch.sigmoid(outputs)
                preds.extend(probs.cpu().numpy().flatten())
                targets.extend(labels.cpu().numpy().flatten())

        avg_loss = running_loss / dataset_size
        logloss = calculate_log_loss(targets, preds)

        return avg_loss, logloss, np.array(preds)

    def _update_bn(self, loader, model):
        """
        Custom update_bn to handle the dual-input (image, angle) architecture.
        Standard torch.optim.swa_utils.update_bn only passes one input.
        """
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

        model.train()
        with torch.no_grad():
            for batch in loader:
                # Handle unpacking safely
                if len(batch) >= 2:
                    images, angles = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                angles = angles.to(self.device)
                model(images, angles)

        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        optimizer,
        scheduler,
        checkpoint_name="checkpoint.pth",
        use_swa=False,
        swa_start_epoch=None,
    ):
        """
        Main training loop handling standard training, validation, and SWA.
        """
        best_loss = float("inf")
        swa_model = None
        swa_scheduler = None

        if use_swa:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
            if swa_start_epoch is None:
                swa_start_epoch = max(1, epochs - Config.SWA_EPOCHS + 1)

        print(f"Starting training for {epochs} epochs. Device: {self.device}")

        for epoch in range(1, epochs + 1):
            in_swa_phase = use_swa and (epoch >= swa_start_epoch)

            # --- Training ---
            train_loss = self.train_one_epoch(train_loader, optimizer)

            # --- Scheduler / SWA Update ---
            if in_swa_phase:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                print(f"Epoch {epoch}/{epochs} [SWA] - Train Loss: {train_loss:.6f}")
            else:
                # Regular scheduler step.
                # If ReduceLROnPlateau, step happens after validation.
                if not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}")

            # --- Validation ---
            if val_loader:
                # Evaluate current base model
                val_loss, val_logloss, _ = self.evaluate(val_loader)
                print(
                    f"Epoch {epoch}/{epochs} - Val Loss: {val_loss:.10f} - Val LogLoss: {val_logloss:.10f}"
                )

                # ReduceLROnPlateau Step
                if (
                    isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    and not in_swa_phase
                ):
                    scheduler.step(val_loss)

                # Save Best Model (Base)
                if not in_swa_phase and val_logloss < best_loss:
                    best_loss = val_logloss
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": self.model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_loss": best_loss,
                        },
                        is_best=True,
                        filename=checkpoint_name,
                    )
            else:
                # No validation (Full training / Replay)
                # Save checkpoint before entering SWA phase
                if use_swa and epoch == swa_start_epoch - 1:
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": self.model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best=False,
                        filename=checkpoint_name,
                    )

        # --- End of Training ---
        if use_swa:
            print("Updating SWA Batch Normalization statistics...")
            self._update_bn(train_loader, swa_model)

            # Save SWA Model
            swa_filename = checkpoint_name.replace(".pth", "_swa.pth")
            save_checkpoint(
                {
                    "epoch": epochs,
                    "state_dict": swa_model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=swa_filename,
            )

            return swa_model.module

        return self.model

    def predict(self, loader, model=None):
        """
        Generates predictions for a given loader.
        """
        eval_model = model if model else self.model
        eval_model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                # Test loader returns (image, angle, id)
                # We only need image and angle
                images, angles = batch[0], batch[1]

                images = images.to(self.device)
                angles = angles.to(self.device)

                outputs = eval_model(images, angles)
                probs = torch.sigmoid(outputs)
                preds.extend(probs.cpu().numpy().flatten())

        return np.array(preds)
