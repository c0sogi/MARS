import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.neural_architecture import OrthogonalSkipGatedMLP
from library.dataset_factory import get_dataloaders
from library.utils import set_seed


class MLPManager:
    """
    Manages the 'Stream B' Neural Network pipeline.
    Handles data loading, model initialization, training with early stopping,
    and inference.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model_path = os.path.join(Config.IDEA_DIR, "best_mlp.pth")
        self.criterion = nn.BCEWithLogitsLoss()
        set_seed(Config.RANDOM_SEED)

    def _init_model(self, metadata_dim):
        """
        Initializes the OrthogonalSkipGatedMLP model.
        """
        model = OrthogonalSkipGatedMLP(metadata_input_dim=metadata_dim)
        model.to(self.device)
        return model

    def _move_batch_to_device(self, inputs, labels=None):
        """
        Moves a batch of data (dictionary of tensors) to the configured device.
        """
        inputs_on_device = {}
        for k, v in inputs.items():
            inputs_on_device[k] = v.to(self.device)

        if labels is not None:
            return inputs_on_device, labels.to(self.device)
        return inputs_on_device

    def train(self, load_cached_data: bool = True):
        """
        Trains the MLP model with Early Stopping.

        Args:
            load_cached_data (bool): Whether to use cached features for dataloaders.

        Returns:
            float: Best Validation ROC AUC score.
        """
        print("\n--- Starting MLP Training ---")

        # 1. Get DataLoaders
        train_loader, val_loader, _, feature_dims = get_dataloaders(
            batch_size=Config.MLP_BATCH_SIZE,
            load_cached_data=load_cached_data,
            num_workers=0,
        )

        # 2. Initialize Model
        metadata_dim = feature_dims["metadata_dim"]
        model = self._init_model(metadata_dim)

        # 3. Setup Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )

        # 4. Training Loop
        best_auc = 0.0
        patience_counter = 0

        print(f"Training on {self.device} for max {Config.MLP_EPOCHS} epochs...")

        for epoch in range(1, Config.MLP_EPOCHS + 1):
            # --- Training Step ---
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for batch_inputs, batch_labels in train_loader:
                batch_inputs, batch_labels = self._move_batch_to_device(
                    batch_inputs, batch_labels
                )

                optimizer.zero_grad()
                logits = model(batch_inputs)
                # logits shape: (B, 1), labels shape: (B,)
                loss = self.criterion(logits.squeeze(), batch_labels)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train_loss = (
                train_loss_sum / train_batches if train_batches > 0 else 0.0
            )

            # --- Validation Step ---
            model.eval()
            val_preds = []
            val_targets = []
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_inputs, batch_labels in val_loader:
                    batch_inputs, batch_labels = self._move_batch_to_device(
                        batch_inputs, batch_labels
                    )

                    logits = model(batch_inputs)
                    loss = self.criterion(logits.squeeze(), batch_labels)

                    val_loss_sum += loss.item()
                    val_batches += 1

                    # Store probabilities for AUC
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    # Handle scalar vs array
                    if np.ndim(probs) == 0:
                        val_preds.append(float(probs))
                    else:
                        val_preds.extend(probs.tolist())

                    val_targets.extend(batch_labels.cpu().numpy().tolist())

            avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

            # Compute AUC
            try:
                current_auc = roc_auc_score(val_targets, val_preds)
            except ValueError:
                # Handle edge case if only one class is present in batch
                current_auc = 0.5

            print(
                f"Epoch {epoch}/{Config.MLP_EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss} | "
                f"Val AUC: {current_auc}"
            )

            # --- Early Stopping ---
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.model_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.MLP_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Best Validation AUC: {best_auc}")
        return best_auc

    def predict_test(self, load_cached_data: bool = True):
        """
        Generates predictions for the test set using the best saved model.

        Args:
            load_cached_data (bool): Whether to use cached features.

        Returns:
            np.ndarray: Predicted probabilities for the test set.
        """
        print("\n--- Generating MLP Predictions for Test Set ---")

        # 1. Get DataLoaders (Test only needed, but factory returns all)
        # We need feature_dims to re-init the model structure correctly
        _, _, test_loader, feature_dims = get_dataloaders(
            batch_size=Config.MLP_BATCH_SIZE,
            load_cached_data=load_cached_data,
            num_workers=0,
        )

        # 2. Initialize Model
        metadata_dim = feature_dims["metadata_dim"]
        model = self._init_model(metadata_dim)

        # 3. Load Weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train first."
            )

        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        # 4. Inference
        all_probs = []

        with torch.no_grad():
            for batch_inputs in test_loader:
                # Test loader returns only inputs (no labels)
                batch_inputs = self._move_batch_to_device(batch_inputs)

                logits = model(batch_inputs)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Handle batch size 1 or scalar output
                if np.ndim(probs) == 0:
                    all_probs.append(float(probs))
                else:
                    all_probs.extend(probs.tolist())

        return np.array(all_probs)
