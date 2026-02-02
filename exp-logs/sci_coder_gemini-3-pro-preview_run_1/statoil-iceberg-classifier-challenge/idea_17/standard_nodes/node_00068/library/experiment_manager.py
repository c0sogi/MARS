import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.configuration import Config
from library.utilities import (
    set_seed,
    save_checkpoint,
    Logger,
    save_submission,
    get_or_create_cached_array,
)
from library.architecture import IcebergResNet
from library.data_loader import get_data_arrays, IcebergDataset
from library.optimization import IcebergLoss, get_optimizer, get_scheduler
from library.training_engine import (
    train_one_epoch,
    validate,
    swa_step,
    update_swa_batch_norm,
)


class ExperimentManager:
    """
    Orchestrates the Two-Phase Training Protocol:
    1. Calibration: Stratified 5-Fold CV to find optimal convergence epoch.
    2. Production: Full-fit training of SWA Ensemble models.
    3. Inference: TTA-based prediction generation.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.logger = Logger()

        # Define transforms locally to ensure consistency across phases
        self.train_transform = A.Compose(
            [
                A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )

        self.val_transform = A.Compose([ToTensorV2()])

    def _create_loader(
        self, images, angles, labels=None, ids=None, mode="train", shuffle=True
    ):
        """Helper to create DataLoaders with correct transforms and settings."""
        dataset = IcebergDataset(
            images,
            angles,
            labels=labels,
            ids=ids,
            transform=self.train_transform if mode == "train" else self.val_transform,
            mode=mode,
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=(mode == "train"),
        )
        return loader

    def run_calibration_phase(self):
        """
        Phase 1: Run Stratified K-Fold CV to determine the global convergence epoch.
        """
        self.logger.log("Starting Phase 1: Calibration (Global Epoch Selection)")

        # Load all training data
        train_imgs, train_angles, train_labels, _, _, _ = get_data_arrays()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

        # Dictionary to store validation losses: epoch -> list of losses
        epoch_losses = {}

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_imgs, train_labels)
        ):
            self.logger.log(f"\n--- Calibration Fold {fold + 1}/5 ---")

            # Prepare data splits
            X_train, X_val = train_imgs[train_idx], train_imgs[val_idx]
            a_train, a_val = train_angles[train_idx], train_angles[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]

            train_loader = self._create_loader(
                X_train, a_train, labels=y_train, mode="train", shuffle=True
            )
            val_loader = self._create_loader(
                X_val, a_val, labels=y_val, mode="val", shuffle=False
            )

            # Initialize Model & Optimization
            model = IcebergResNet().to(self.device)
            optimizer = get_optimizer(model)
            scheduler = get_scheduler(optimizer)
            criterion = IcebergLoss()

            best_fold_loss = float("inf")
            patience_counter = 0

            for epoch in range(1, Config.MAX_EPOCHS_PHASE_1 + 1):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, self.device, epoch
                )
                val_loss = validate(model, val_loader, self.device)

                self.logger.log_metrics(
                    epoch,
                    {"Train Loss": train_loss, "Val Loss": val_loss},
                    phase=f"Fold {fold+1}",
                )

                scheduler.step(val_loss)

                # Record loss for global averaging
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(val_loss)

                # Early Stopping Logic
                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    self.logger.log(f"Early stopping triggered at epoch {epoch}")
                    break

        # Determine Convergence Epoch
        # Calculate mean loss for each epoch where data is available
        avg_losses = {e: np.mean(losses) for e, losses in epoch_losses.items()}

        # Find epoch with minimum average loss
        best_epoch = min(avg_losses, key=avg_losses.get)
        min_loss = avg_losses[best_epoch]

        self.logger.log(
            f"\nCalibration Result: Convergence Epoch (E_conv) = {best_epoch} with Avg Val Loss = {min_loss}"
        )
        return best_epoch

    def run_production_phase(self, convergence_epoch):
        """
        Phase 2: Train Ensemble Models on Full Data using SWA.
        """
        self.logger.log(
            f"\nStarting Phase 2: Production (Full-Fit SWA) for {Config.NUM_ENSEMBLE_MODELS} models"
        )

        # Load all training data
        train_imgs, train_angles, train_labels, _, _, _ = get_data_arrays()

        # Create Full Training Loader
        full_train_loader = self._create_loader(
            train_imgs, train_angles, labels=train_labels, mode="train", shuffle=True
        )

        trained_models = []

        for i in range(Config.NUM_ENSEMBLE_MODELS):
            self.logger.log(f"\n--- Training Ensemble Model {i + 1} ---")

            # Set distinct seed for diversity
            set_seed(Config.SEED + i)

            model = IcebergResNet().to(self.device)
            optimizer = get_optimizer(model)
            scheduler = get_scheduler(optimizer)
            criterion = IcebergLoss()

            # 1. Standard Training Phase
            self.logger.log(
                f"Phase 2a: Standard Training for {convergence_epoch} epochs"
            )
            for epoch in range(1, convergence_epoch + 1):
                loss = train_one_epoch(
                    model, full_train_loader, optimizer, criterion, self.device, epoch
                )
                # Step scheduler using train loss as proxy since we have no validation set
                scheduler.step(loss)

            # 2. SWA Training Phase
            self.logger.log(f"Phase 2b: SWA Training for {Config.SWA_EPOCHS} epochs")
            swa_model = AveragedModel(model)

            # Reset Learning Rate for SWA
            for param_group in optimizer.param_groups:
                param_group["lr"] = Config.SWA_LR

            for swa_ep in range(Config.SWA_EPOCHS):
                current_epoch = convergence_epoch + swa_ep + 1
                train_one_epoch(
                    model,
                    full_train_loader,
                    optimizer,
                    criterion,
                    self.device,
                    current_epoch,
                )
                swa_step(swa_model, model)

            # Update Batch Norm Statistics
            self.logger.log("Updating SWA Batch Normalization statistics...")
            update_swa_batch_norm(swa_model, full_train_loader, self.device)

            # Save Checkpoint
            save_filename = f"ensemble_{i}_swa.pth"
            save_checkpoint(swa_model.state_dict(), False, filename=save_filename)
            trained_models.append(swa_model)

        return trained_models

    def run_inference(self, models):
        """
        Generate predictions using TTA and Ensemble Averaging.
        """
        self.logger.log("\nStarting Inference with TTA")

        # Load test data
        _, _, _, test_imgs, test_angles, test_ids = get_data_arrays()

        # Test Loader
        test_loader = self._create_loader(
            test_imgs, test_angles, ids=test_ids, mode="test", shuffle=False
        )

        final_probs = []
        all_ids = []

        # Set all models to eval mode
        for model in models:
            model.eval()

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)

                # Accumulator for probabilities
                batch_probs_sum = torch.zeros(images.size(0), device=self.device)

                # TTA Views: Original, H-Flip, V-Flip
                views = [
                    images,
                    torch.flip(images, [3]),  # Horizontal Flip (W is dim 3)
                    torch.flip(images, [2]),  # Vertical Flip (H is dim 2)
                ]

                # Aggregate predictions
                for model in models:
                    for view in views:
                        logits = model(view, angles)
                        probs = torch.sigmoid(logits).view(-1)
                        batch_probs_sum += probs

                # Compute Average
                # Total Count = Num_Models * Num_Views
                count = len(models) * len(views)
                avg_probs = batch_probs_sum / count

                final_probs.extend(avg_probs.cpu().numpy())
                all_ids.extend(ids)

        # Save Submission
        save_submission(all_ids, final_probs, "submission.csv")
        self.logger.log("Submission generated successfully.")

    def execute(self):
        """
        Main execution entry point.
        """
        # Step 1: Calibration
        e_conv = self.run_calibration_phase()

        # Step 2: Production Training
        models = self.run_production_phase(e_conv)

        # Step 3: Inference
        self.run_inference(models)
