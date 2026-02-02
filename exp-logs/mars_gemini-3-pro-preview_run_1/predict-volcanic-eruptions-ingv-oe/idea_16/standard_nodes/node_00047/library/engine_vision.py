import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import VolcanoDataset
from library.model_vision import VolcanoEfficientNet
from library.spectrogram_generator import SpectrogramProcessor


class VisionTrainer:
    """
    Manages the training, validation, and inference lifecycle for the Vision Branch (EfficientNet).
    Handles 5-Fold CV, OOF generation, and submission creation.
    """

    def __init__(self):
        self.config = Config
        self.device = self.config.DEVICE

        # Ensure reproducibility
        seed_everything(self.config.SEED)

        # Create necessary directories
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

    def _ensure_spectrograms(self):
        """
        Ensures that spectrograms for all datasets are generated and cached.
        """
        print("Checking and generating spectrograms if needed...")
        processor = SpectrogramProcessor()
        # Generate for Train, Val, and Test
        # We process all to ensure no missing files during CV or Inference
        processor.generate_dataset(self.config.TRAIN_METADATA_PATH)
        processor.generate_dataset(self.config.VAL_METADATA_PATH)
        processor.generate_dataset(self.config.TEST_METADATA_PATH)

    def train_epoch(self, model, loader, optimizer, criterion, scheduler):
        """
        Runs one epoch of training.
        """
        model.train()
        running_loss = 0.0

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            # Targets from dataset are log-scaled. Add channel dim: (Batch,) -> (Batch, 1)
            targets = targets.to(self.device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Step scheduler after epoch
        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(loader.dataset)
        return epoch_loss

    def validate(self, model, loader, criterion):
        """
        Runs validation, calculates loss on log-scale and MAE on original scale.
        """
        model.eval()
        running_loss = 0.0
        preds_linear = []
        actuals_linear = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets_gpu = targets.to(self.device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets_gpu)

                running_loss += loss.item() * inputs.size(0)

                # Inverse Transform: expm1 to get back to time_to_eruption
                # Output shape (Batch, 1) -> flatten
                batch_preds = torch.expm1(outputs).cpu().numpy().flatten()
                batch_actuals = torch.expm1(targets_gpu).cpu().numpy().flatten()

                preds_linear.extend(batch_preds)
                actuals_linear.extend(batch_actuals)

        epoch_loss = running_loss / len(loader.dataset)

        # Calculate MAE on the original scale (the competition metric)
        mae = get_score(np.array(actuals_linear), np.array(preds_linear))

        return epoch_loss, mae, np.array(preds_linear), np.array(actuals_linear)

    def predict(self, model, loader):
        """
        Generates predictions for a dataset (e.g., Test).
        Returns segment_ids and predicted values (linear scale).
        """
        model.eval()
        preds = []
        ids = []

        with torch.no_grad():
            for inputs, segment_ids in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                batch_preds = torch.expm1(outputs).cpu().numpy().flatten()

                preds.extend(batch_preds)
                ids.extend(segment_ids.numpy().flatten())

        return ids, preds

    def run_cv(self, debug_sample=None):
        """
        Executes the 5-Fold Cross-Validation loop.

        Args:
            debug_sample (int, optional): If set, limits the dataset size for debugging.
        """
        # 1. Prepare Data
        self._ensure_spectrograms()

        # Load and combine provided Train and Val metadata to form the full development set
        df_train = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(self.config.VAL_METADATA_PATH)
        df_full = pd.concat([df_train, df_val], ignore_index=True)

        if debug_sample:
            df_full = df_full.head(debug_sample)
            print(f"Debug Mode: Training on {len(df_full)} samples.")

        # 2. Setup KFold
        kf = KFold(
            n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED
        )

        # Arrays to store OOF predictions
        oof_preds = np.zeros(len(df_full))
        oof_targets = np.zeros(len(df_full))
        segment_ids_full = df_full[self.config.SEGMENT_ID_COL].values

        # List to store test predictions from each fold
        test_preds_folds = []

        # 3. Setup Test Loader (Once)
        test_dataset = VolcanoDataset(
            self.config.TEST_METADATA_PATH, is_test=True, debug_sample=debug_sample
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # 4. CV Loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_full)):
            print(f"\n{'='*20} Fold {fold} {'='*20}")

            # Create temporary metadata files for this fold
            # This allows us to reuse the file-based VolcanoDataset class
            fold_train_df = df_full.iloc[train_idx]
            fold_val_df = df_full.iloc[val_idx]

            fold_train_path = os.path.join(
                self.config.WORKING_DIR, f"train_fold_{fold}.csv"
            )
            fold_val_path = os.path.join(
                self.config.WORKING_DIR, f"val_fold_{fold}.csv"
            )

            fold_train_df.to_csv(fold_train_path, index=False)
            fold_val_df.to_csv(fold_val_path, index=False)

            # Create DataLoaders
            train_ds = VolcanoDataset(fold_train_path, is_test=False)
            val_ds = VolcanoDataset(fold_val_path, is_test=False)

            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            model = VolcanoEfficientNet(pretrained=True)
            model.to(self.device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.EPOCHS, eta_min=1e-6
            )

            # Loss Function (MAE on Log Scale)
            criterion = nn.L1Loss()

            # Training Loop Variables
            best_mae = float("inf")
            best_model_path = os.path.join(
                self.config.WORKING_DIR, f"vision_model_fold_{fold}.pth"
            )
            patience = 8
            patience_counter = 0

            for epoch in range(self.config.EPOCHS):
                train_loss = self.train_epoch(
                    model, train_loader, optimizer, criterion, scheduler
                )
                val_loss, val_mae, _, _ = self.validate(model, val_loader, criterion)

                print(
                    f"Epoch {epoch+1}/{self.config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
                )

                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # 5. Inference for this Fold
            print(f"Loading best model for Fold {fold}...")
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))

            # Generate OOF Preds
            _, _, val_preds, val_actuals = self.validate(model, val_loader, criterion)
            oof_preds[val_idx] = val_preds
            oof_targets[val_idx] = val_actuals

            # Generate Test Preds
            _, test_preds = self.predict(model, test_loader)
            test_preds_folds.append(test_preds)

        # 6. Save Results
        # Save OOF
        oof_df = pd.DataFrame(
            {"segment_id": segment_ids_full, "time_to_eruption": oof_preds}
        )
        oof_path = os.path.join(self.config.WORKING_DIR, "vision_oof.csv")
        oof_df.to_csv(oof_path, index=False)

        overall_mae = get_score(oof_targets, oof_preds)
        print(f"\nOverall CV MAE: {overall_mae}")
        print(f"OOF predictions saved to {oof_path}")

        # Average Test Predictions
        avg_test_preds = np.mean(test_preds_folds, axis=0)

        # Get Test IDs (from last prediction call)
        test_ids, _ = self.predict(model, test_loader)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": avg_test_preds}
        )

        # Save Intermediate Test Preds
        test_pred_path = os.path.join(self.config.WORKING_DIR, "vision_test.csv")
        sub_df.to_csv(test_pred_path, index=False)
        print(f"Test predictions saved to {test_pred_path}")

        # Save Final Submission
        final_sub_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(final_sub_path, index=False)
        print(f"Submission saved to {final_sub_path}")
