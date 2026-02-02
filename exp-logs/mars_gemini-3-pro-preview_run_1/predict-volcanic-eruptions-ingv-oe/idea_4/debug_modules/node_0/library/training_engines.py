import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.utils import seed_everything, calculate_mae
from library.model_definitions import SeismicEfficientNet


class CNNTrainer:
    """
    Trainer for the 2D-CNN Spectrogram Model (Branch B).
    Handles training, validation (with inverse target scaling), and prediction.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(Config.DEVICE)
        self.working_dir = Config.WORKING_DIR
        self.model_name = "cnn_model.pth"

        # Hyperparameters
        self.epochs = Config.CNN_EPOCHS
        self.lr = Config.CNN_LR
        self.weight_decay = Config.CNN_WEIGHT_DECAY
        self.patience = Config.CNN_PATIENCE

    def train(self, train_loader, val_loader, fold=0):
        """
        Trains the CNN model with early stopping.
        """
        seed_everything(Config.SEED + fold)

        # Initialize Model
        model = SeismicEfficientNet(pretrained=True)
        model.to(self.device)

        # Loss function: L1 Loss (MAE) on scaled targets
        criterion = nn.L1Loss()

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        best_val_mae = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.working_dir, f"cnn_fold_{fold}.pth")

        print(f"Starting CNN training for Fold {fold}...")

        for epoch in range(self.epochs):
            # --- Training ---
            model.train()
            train_loss_sum = 0.0
            train_steps = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)  # (Batch, 1)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_steps += 1

            avg_train_loss = train_loss_sum / max(1, train_steps)

            # --- Validation ---
            val_mae = self.validate(model, val_loader)

            # Update Scheduler
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss (Scaled): {avg_train_loss:.6f} - "
                f"Val MAE (Real): {val_mae:.6f}"
            )

            # --- Early Stopping ---
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Best Val MAE for Fold {fold}: {best_val_mae}")
        return best_val_mae

    def validate(self, model, val_loader):
        """
        Evaluates the model on the validation set.
        Performs inverse transformation on predictions to calculate real MAE.
        """
        model.eval()
        preds = []
        actuals = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                # Collect results
                preds.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())

        preds = np.concatenate(preds).flatten()
        actuals = np.concatenate(actuals).flatten()

        # Inverse Transform
        if Config.TARGET_SCALING == "log1p":
            preds_real = np.expm1(preds)
            actuals_real = np.expm1(actuals)
        else:
            preds_real = preds
            actuals_real = actuals

        # Clip negative predictions to 0 (time cannot be negative)
        preds_real = np.maximum(preds_real, 0)

        return mean_absolute_error(actuals_real, preds_real)

    def predict(self, test_loader, fold=0):
        """
        Generates predictions using the trained model from a specific fold.
        """
        model_path = os.path.join(self.working_dir, f"cnn_fold_{fold}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = SeismicEfficientNet(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        preds = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                preds.append(outputs.cpu().numpy())

        preds = np.concatenate(preds).flatten()

        # Inverse Transform
        if Config.TARGET_SCALING == "log1p":
            preds_real = np.expm1(preds)
        else:
            preds_real = preds

        return np.maximum(preds_real, 0)


class LGBMTrainer:
    """
    Trainer for the LightGBM Model (Branch A).
    Handles tabular data training with early stopping.
    """

    def __init__(self):
        self.params = Config.LGB_PARAMS.copy()
        self.working_dir = Config.WORKING_DIR
        self.early_stopping_rounds = Config.LGB_EARLY_STOPPING_ROUNDS

    def train(self, X_train, y_train, X_val, y_val, fold=0):
        """
        Trains LightGBM model.
        """
        seed_everything(Config.SEED + fold)

        # Create Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]

        print(f"Starting LightGBM training for Fold {fold}...")

        model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Save model
        model_path = os.path.join(self.working_dir, f"lgb_fold_{fold}.txt")
        model.save_model(model_path)

        # Get best score
        # LightGBM metrics are stored in model.best_score
        # Structure: {'train': {'l1': val}, 'valid': {'l1': val}}
        best_score = model.best_score["valid"]["l1"]
        print(f"Best Val MAE for Fold {fold}: {best_score}")

        return model, best_score

    def predict(self, X_test, fold=0):
        """
        Generates predictions using the trained model.
        """
        model_path = os.path.join(self.working_dir, f"lgb_fold_{fold}.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        return np.maximum(preds, 0)


class RidgeStacker:
    """
    Meta-Learner for Stacking (Ensemble).
    Combines OOF predictions from CNN and LightGBM using Ridge Regression.
    """

    def __init__(self):
        self.alpha = Config.META_ALPHA
        self.working_dir = Config.WORKING_DIR
        self.model = None

    def fit(self, X_meta, y):
        """
        Trains the Ridge regressor.
        X_meta: Matrix of shape (N_samples, 2) -> [LGBM_Preds, CNN_Preds]
        y: True targets
        """
        print("Training Meta-Learner (Ridge)...")
        self.model = Ridge(alpha=self.alpha, random_state=Config.SEED)
        self.model.fit(X_meta, y)

        # Print coefficients to see contribution of each model
        print(f"Meta-Learner Coefficients: {self.model.coef_}")
        print(f"Meta-Learner Intercept: {self.model.intercept_}")

    def predict(self, X_meta):
        """
        Predicts final output.
        """
        if self.model is None:
            raise RuntimeError("Meta-learner has not been trained yet.")

        preds = self.model.predict(X_meta)
        return np.maximum(preds, 0)
