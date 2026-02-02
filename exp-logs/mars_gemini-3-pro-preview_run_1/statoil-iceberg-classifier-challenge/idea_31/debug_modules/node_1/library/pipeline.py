import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn

from library.config import Config
from library.utils import set_seed, get_logger
from library.data import get_kfold_loaders, get_production_loader, get_test_loader
from library.model import IsovariantResNet18
from library.engine import train_one_epoch, train_swa_epoch, evaluate_tta, predict_tta


class LabelSmoothingBCE(nn.Module):
    def __init__(self, epsilon=0.05):
        super(LabelSmoothingBCE, self).__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # target shape: (batch_size, 1)
        smooth_target = target * (1 - self.epsilon) + 0.5 * self.epsilon
        return self.bce(pred, smooth_target)


class Pipeline:
    def __init__(self):
        self.logger = get_logger("pipeline")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Config.setup_dirs()

    def run_calibration_phase(self):
        """
        Phase 1: Calibration (Step Volume Discovery)
        Runs 5-Fold CV to determine the optimal number of epochs (E_opt).
        """
        self.logger.info("Starting Phase 1: Calibration (5-Fold CV)")

        # Store validation losses: [fold_idx][epoch_idx] -> loss
        # We assume max epochs is enough to cover convergence
        max_epochs = Config.MAX_EPOCHS_PHASE1
        fold_losses = {fold: [] for fold in range(5)}

        for fold in range(5):
            self.logger.info(f"--- Fold {fold} ---")
            set_seed(Config.SEED + fold)

            # Data
            train_loader, val_loader = get_kfold_loaders(fold, n_splits=5)

            # Model
            model = IsovariantResNet18().to(self.device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR_BASE, weight_decay=Config.WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=Config.SCHEDULER_FACTOR,
                patience=Config.SCHEDULER_PATIENCE,
                min_lr=Config.MIN_LR,
            )

            # Loss
            criterion = LabelSmoothingBCE(epsilon=Config.LABEL_SMOOTHING)

            best_fold_loss = float("inf")
            early_stop_counter = 0

            for epoch in range(max_epochs):
                # Train
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, self.device
                )

                # Validate (TTA)
                val_loss, val_acc = evaluate_tta(
                    model, val_loader, criterion, self.device
                )

                # Step Scheduler
                scheduler.step(val_loss)

                # Record history
                fold_losses[fold].append(val_loss)

                self.logger.info(
                    f"Fold {fold} Ep {epoch+1}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, Acc {val_acc:.6f}"
                )

                # Early Stopping Check
                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Clean up
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

        # Determine E_opt
        # We calculate the average validation loss across folds for each epoch.
        # Since folds might have different lengths due to early stopping, we truncate to the minimum length
        # or pad. A safer approach for "optimal stopping" is to look at the curve.
        # Given the prompt instruction: "Identify the epoch E_opt that minimizes the TTA-Validation Loss"
        # We will find the epoch that minimizes the average loss across available folds.

        min_len = min(len(losses) for losses in fold_losses.values())
        avg_losses = []
        for i in range(min_len):
            avg_loss = np.mean([fold_losses[f][i] for f in range(5)])
            avg_losses.append(avg_loss)

        best_epoch_idx = np.argmin(avg_losses)
        e_opt = int(best_epoch_idx) + 1  # 1-based

        self.logger.info(
            f"Phase 1 Complete. Optimal Epoch (E_opt): {e_opt} (Min Avg Val Loss: {avg_losses[best_epoch_idx]:.6f})"
        )
        return e_opt

    def run_production_phase(self, e_opt):
        """
        Phase 2: Production (Isovariant Full-Fit)
        Trains 5 independent models on 100% data using scaled epochs and SWA.
        """
        # Isovariant Scaling
        e_prod = math.ceil(e_opt * Config.ISOVARIANT_SCALE)
        self.logger.info(
            f"Starting Phase 2: Production. E_prod = ceil({e_opt} * {Config.ISOVARIANT_SCALE}) = {e_prod}"
        )

        # We train 5 ensemble members
        num_models = 5

        for i in range(num_models):
            self.logger.info(f"--- Training Production Model {i} ---")
            set_seed(Config.SEED + i * 10)  # Distinct seeds for ensemble diversity

            # Full Data
            train_loader = get_production_loader()

            # Model
            model = IsovariantResNet18().to(self.device)

            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR_BASE, weight_decay=Config.WEIGHT_DECAY
            )

            # Deterministic Schedule (Cosine)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=e_prod, eta_min=Config.LR_SWA
            )

            # Loss
            criterion = LabelSmoothingBCE(epsilon=Config.LABEL_SMOOTHING)

            # 1. Main Training Phase
            for epoch in range(e_prod):
                loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, self.device
                )
                scheduler.step()
                self.logger.info(
                    f"Model {i} Phase 1 Ep {epoch+1}/{e_prod}: Loss {loss:.6f}, LR {optimizer.param_groups[0]['lr']:.2e}"
                )

            # 2. SWA Phase
            self.logger.info(
                f"Model {i} Entering SWA Phase for {Config.SWA_EPOCHS} epochs..."
            )
            swa_model = AveragedModel(model)
            swa_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.SWA_EPOCHS, eta_min=Config.LR_SWA
            )

            # Fix LR for SWA (or use a very tight cycle, prompt says constant 1e-5)
            # We explicitly set param groups to LR_SWA
            for param_group in optimizer.param_groups:
                param_group["lr"] = Config.LR_SWA

            for swa_ep in range(Config.SWA_EPOCHS):
                # Train one epoch
                loss = train_swa_epoch(
                    model, train_loader, optimizer, criterion, self.device
                )
                # Update SWA
                swa_model.update_parameters(model)
                self.logger.info(
                    f"Model {i} SWA Ep {swa_ep+1}/{Config.SWA_EPOCHS}: Loss {loss:.6f}"
                )

            # 3. Update BN Statistics
            self.logger.info(f"Model {i} Updating BN statistics...")
            update_bn(train_loader, swa_model, device=self.device)

            # 4. Save
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
            torch.save(swa_model.state_dict(), save_path)
            self.logger.info(f"Saved SWA model to {save_path}")

            # Cleanup
            del model, swa_model, optimizer, scheduler, train_loader
            torch.cuda.empty_cache()

    def generate_submission(self):
        """
        Loads SWA models, predicts on test set using TTA, and saves submission.
        """
        self.logger.info("Generating Submission...")

        # Load Test Data
        test_loader, test_ids = get_test_loader()

        # Load Models
        models = []
        for i in range(5):
            path = os.path.join(Config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
            if not os.path.exists(path):
                self.logger.warning(f"Checkpoint {path} not found. Skipping.")
                continue

            model = IsovariantResNet18().to(self.device)
            # SWA saves the module wrapper, so we might need to be careful with keys
            # AveragedModel saves keys as 'module.layer...', IsovariantResNet18 expects 'layer...'
            # However, AveragedModel usually wraps the model.
            # Let's load state dict and handle 'module.' prefix if present (AveragedModel adds it)
            state_dict = torch.load(path, map_location=self.device)

            # Fix keys if wrapped in AveragedModel (which uses 'module.')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k.startswith("n_averaged"):
                    continue  # Skip SWA counter
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            model.eval()
            models.append(model)

        if not models:
            raise RuntimeError("No models loaded for submission!")

        self.logger.info(f"Loaded {len(models)} models for ensemble.")

        # Inference
        ensemble_preds = []
        for model in models:
            preds = predict_tta(model, test_loader, self.device)
            ensemble_preds.append(preds)

        # Average
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Save
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        # Ensure formatting matches sample (id, is_iceberg)
        # Rounding is not required by prompt, but usually 6-8 decimals is good.
        # The prompt says "probability ... number between 0 and 1".

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
