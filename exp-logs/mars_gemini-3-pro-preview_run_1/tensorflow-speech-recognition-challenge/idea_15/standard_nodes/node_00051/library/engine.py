import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import (
    MODEL_PARAMS,
    TRAINING_PARAMS,
    WORKING_DIR,
    SUBMISSION_PATH,
)
from library.utils import (
    AverageMeter,
    calculate_accuracy,
    save_submission,
    set_seed,
)
from library.model import DilatedEfficientNet
from library.swa_utils import SWAHandler


class Trainer:
    """
    Manages the training, validation, SWA pipeline, and inference.
    """

    def __init__(self, train_loader, val_loader, test_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize Model
        self.model = DilatedEfficientNet(config=MODEL_PARAMS).to(self.device)

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_PARAMS["learning_rate"],
            weight_decay=TRAINING_PARAMS["weight_decay"],
        )

        # Scheduler (Cosine Annealing for Phase 1)
        # We set T_max to the start of SWA to ensure it decays fully before switching strategies
        self.swa_start_epoch = TRAINING_PARAMS["swa_start_epoch"]
        self.total_epochs = TRAINING_PARAMS["epochs"]

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.swa_start_epoch,
            eta_min=TRAINING_PARAMS["min_lr"],
        )

        # SWA Handler
        self.swa_handler = SWAHandler(self.model, device=self.device)
        self.swa_lr = TRAINING_PARAMS["swa_lr"]

        # Tracking
        self.best_acc = 0.0
        self.best_model_path = os.path.join(WORKING_DIR, "best_model_phase1.pth")
        self.swa_model_path = os.path.join(WORKING_DIR, "best_model_swa.pth")

    def _mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_one_epoch(self, epoch, is_swa_phase=False):
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        # Set constant LR if in SWA phase
        if is_swa_phase:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.swa_lr

        current_lr = self.optimizer.param_groups[0]["lr"]

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup
            images, labels_a, labels_b, lam = self._mixup_data(
                images, labels, TRAINING_PARAMS["mixup_alpha"]
            )

            # Forward pass
            outputs = self.model(images)
            loss = self._mixup_criterion(
                self.criterion, outputs, labels_a, labels_b, lam
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Measure accuracy (using the stronger label for approximation)
            acc = calculate_accuracy(outputs, labels)

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

        print(
            f"Epoch [{epoch+1}/{self.total_epochs}] Train Loss: {losses.avg:.10f} | Train Acc: {accuracies.avg:.10f} | LR: {current_lr:.8f}"
        )
        return losses.avg, accuracies.avg

    def evaluate(self, loader, model_to_eval=None):
        if model_to_eval is None:
            model_to_eval = self.model

        model_to_eval.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model_to_eval(images)
                loss = self.criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)

                losses.update(loss.item(), images.size(0))
                accuracies.update(acc, images.size(0))

        return losses.avg, accuracies.avg

    def fit(self):
        print(f"Starting training on device: {self.device}")
        set_seed(TRAINING_PARAMS["seed"])

        # ---------------------------------------------------------------------
        # Training Loop
        # ---------------------------------------------------------------------
        for epoch in range(self.total_epochs):
            # Determine Phase
            # swa_start_epoch is 1-based in config usually, but let's assume 0-indexed logic
            # If config says start at 41 (1-based), that is index 40.
            is_swa_phase = epoch >= (self.swa_start_epoch - 1)

            # Train
            self.train_one_epoch(epoch, is_swa_phase)

            # Phase 1: Standard Validation and Scheduler
            if not is_swa_phase:
                val_loss, val_acc = self.evaluate(self.val_loader)
                print(
                    f"Epoch [{epoch+1}/{self.total_epochs}] Val Loss: {val_loss:.10f} | Val Acc: {val_acc:.10f}"
                )

                # Save Best Phase 1 Model
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"New best model saved with accuracy: {self.best_acc:.10f}")

                # Step Scheduler
                self.scheduler.step()

            # Phase 2: SWA Collection
            else:
                print(
                    f"Epoch [{epoch+1}/{self.total_epochs}] SWA Phase: Updating averaged model."
                )
                self.swa_handler.update_average(self.model)
                # We do not step the scheduler here; LR is held constant manually in train_one_epoch

        # ---------------------------------------------------------------------
        # Post-Training: SWA Finalization
        # ---------------------------------------------------------------------
        if self.swa_start_epoch <= self.total_epochs:
            print("\nTraining finished. Finalizing SWA Model...")

            # Update BN Statistics for SWA Model
            print(
                "Updating Batch Normalization statistics for SWA model (this may take a while)..."
            )
            self.swa_handler.update_bn_statistics(self.train_loader)

            # Save SWA Model
            self.swa_handler.save_model(self.swa_model_path)
            print(f"SWA model saved to {self.swa_model_path}")

            # Evaluate SWA Model
            swa_model = self.swa_handler.get_averaged_model()
            val_loss, val_acc = self.evaluate(self.val_loader, model_to_eval=swa_model)
            print(
                f"Final SWA Model Validation - Loss: {val_loss:.10f} | Acc: {val_acc:.10f}"
            )
        else:
            print("\nSWA skipped (swa_start_epoch > total_epochs).")

    def predict_and_submit(self):
        """
        Generates predictions using the final SWA model and saves to CSV.
        """
        print("\nGenerating predictions for test set...")

        # Use SWA model for inference
        model = self.swa_handler.get_averaged_model()
        model.eval()

        predictions = []
        filenames = []

        # Extract filenames from dataset
        # The dataset returns (image, label_idx), but we need filenames for submission.
        # We access the underlying dataframe of the test dataset.
        test_df = self.test_loader.dataset.df

        with torch.no_grad():
            for i, (images, _) in enumerate(self.test_loader):
                images = images.to(self.device)

                # Forward
                outputs = model(images)

                # Get predictions
                _, preds = torch.max(outputs, 1)

                predictions.extend(preds.cpu().numpy())

        # Get filenames corresponding to the order in DataLoader
        # Since shuffle=False for test_loader, order is preserved.
        filenames = test_df["filepath"].apply(os.path.basename).tolist()

        if len(filenames) != len(predictions):
            print(
                f"Warning: Number of filenames ({len(filenames)}) does not match predictions ({len(predictions)})"
            )

        # Save
        save_submission(predictions, filenames, SUBMISSION_PATH)
        print(f"Submission saved to {SUBMISSION_PATH}")
