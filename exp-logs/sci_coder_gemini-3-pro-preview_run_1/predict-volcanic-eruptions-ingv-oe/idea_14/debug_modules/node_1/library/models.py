import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import lightgbm as lgb
from library.config import Config


class ScalarFusedEfficientNet(nn.Module):
    """
    EfficientNet-B0 modified to accept 20 input channels (Dual-Resolution Spectrograms)
    and fuse normalized scalar features (Energy Stats) before the final classification layer.
    """

    def __init__(self):
        super(ScalarFusedEfficientNet, self).__init__()
        self.config = Config

        # 1. Load Backbone
        # efficientnet_b0 output features: 1280
        # in_chans=20: timm handles weight recycling/adaptation for the first layer
        self.backbone = timm.create_model(
            self.config.MODEL_NAME,
            pretrained=True,
            in_chans=self.config.IN_CHANNELS,
            num_classes=0,  # Remove classifier to get pooling output (embedding)
        )

        # Get the number of features from the backbone (usually 1280 for B0)
        self.num_features = self.backbone.num_features

        # 2. Normalized Scalar Fusion Components
        # Batch Normalization for the scalar vector to handle scale mismatch
        # Input dimension: 30 (10 sensors * 3 stats)
        self.scalar_bn = nn.BatchNorm1d(self.config.SCALAR_DIM)

        # 3. Final Regression Head
        # Concatenate backbone features + normalized scalars
        self.fc = nn.Linear(self.num_features + self.config.SCALAR_DIM, 1)

    def forward(self, images, scalars):
        """
        Args:
            images: (Batch, 20, 128, 128)
            scalars: (Batch, 30)
        """
        # Extract image embeddings
        # Shape: (Batch, num_features)
        img_embedding = self.backbone(images)

        # Normalize scalars
        # Shape: (Batch, 30)
        scalars_norm = self.scalar_bn(scalars)

        # Concatenate
        # Shape: (Batch, num_features + 30)
        combined = torch.cat([img_embedding, scalars_norm], dim=1)

        # Predict
        # Shape: (Batch, 1)
        output = self.fc(combined)

        return output


class LightGBMWrapper:
    """
    Wrapper for LightGBM training and inference to ensure consistent configuration
    and interface for the stacking ensemble.
    """

    def __init__(self):
        self.config = Config
        self.model = None
        self.params = self.config.LGB_PARAMS.copy()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation targets.
        """
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Callbacks for logging and early stopping
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.LGB_EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Print final metric
        if X_val is not None:
            # Best iteration is handled by early_stopping callback automatically loading best model
            preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
            mae = np.mean(np.abs(y_val - preds))
            print(f"LGBM Final Validation MAE: {mae}")

    def predict(self, X):
        """
        Predicts using the trained model.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.array: Predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save_model(self, path):
        """Saves the model to a text file."""
        if self.model is not None:
            self.model.save_model(path)

    def load_model(self, path):
        """Loads the model from a text file."""
        if os.path.exists(path):
            self.model = lgb.Booster(model_file=path)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
