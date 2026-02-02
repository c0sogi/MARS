import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.metrics import roc_auc_score

from library.config import MLP_PARAMS, DEVICE, RANDOM_STATE
from library.utils import set_seed
from library.mlp_architecture import SkipGatedMLP
from library.mlp_data import create_dataloader


class MLPTrainer:
    """
    Trainer class for the SkipGatedMLP model.
    Manages the training loop, validation, early stopping, and inference.
    """

    def __init__(self, params=None):
        """
        Initialize the trainer.

        Args:
            params (dict, optional): Hyperparameters for the model and training.
                                     Defaults to MLP_PARAMS from config.
        """
        self.params = params if params is not None else MLP_PARAMS.copy()
        self.model = None
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    def _to_tensor_dict(self, batch_dict):
        """Moves a dictionary of tensors to the configured device."""
        return {k: v.to(self.device) for k, v in batch_dict.items()}

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        max_samples=None,
        epochs=None,
        batch_size=None,
    ):
        """
        Trains the MLP model using the provided data.

        Args:
            X_train (dict): Dictionary of training features.
            y_train (array-like): Training targets.
            X_val (dict, optional): Dictionary of validation features.
            y_val (array-like, optional): Validation targets.
            max_samples (int, optional): Limit the number of samples for debugging.
            epochs (int, optional): Override the number of epochs from params.
            batch_size (int, optional): Override the batch size from params.
        """
        set_seed(RANDOM_STATE)

        # Hyperparameter Overrides
        current_epochs = epochs if epochs is not None else self.params["epochs"]
        current_batch_size = (
            batch_size if batch_size is not None else self.params["batch_size"]
        )

        # Infer input dimensions from the data
        # Assumes X_train contains the keys 'title_emb' and 'dense_features'
        input_embedding_dim = X_train["title_emb"].shape[1]
        dense_input_dim = X_train["dense_features"].shape[1]

        # Initialize Model
        self.model = SkipGatedMLP(
            input_embedding_dim=input_embedding_dim,
            dense_input_dim=dense_input_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout_prob=self.params["dropout_prob"],
            dropout_dense=self.params["dropout_dense"],
        ).to(self.device)

        # Create DataLoaders
        train_loader = create_dataloader(
            X_train,
            y_train,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=0,
            max_samples=max_samples,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_dataloader(
                X_val,
                y_val,
                batch_size=current_batch_size * 2,
                shuffle=False,
                num_workers=0,
                max_samples=max_samples,
            )

        # Optimizer, Loss, and Scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            verbose=False,
        )

        # Training Loop Variables
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(current_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = self._to_tensor_dict(batch_X)
                # Reshape targets to (B, 1) for BCEWithLogitsLoss
                batch_y = batch_y.float().to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Step
            if val_loader:
                self.model.eval()
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = self._to_tensor_dict(batch_X)
                        logits = self.model(batch_X)
                        probs = torch.sigmoid(logits)

                        val_preds.extend(probs.cpu().numpy().flatten())
                        val_targets.extend(batch_y.numpy().flatten())

                val_auc = roc_auc_score(val_targets, val_preds)
                print(
                    f"Epoch {epoch+1}/{current_epochs} - Train Loss: {avg_train_loss} - Val AUC: {val_auc}"
                )

                # Scheduler Step
                scheduler.step(val_auc)

                # Early Stopping Logic
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.params["patience"]:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                # If no validation set, just print loss and save state
                print(
                    f"Epoch {epoch+1}/{current_epochs} - Train Loss: {avg_train_loss}"
                )
                best_model_state = copy.deepcopy(self.model.state_dict())

        # Restore best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f"MLP Training Complete. Best Validation AUC: {best_val_auc}")

    def predict(self, X):
        """
        Generates predictions for the given features.

        Args:
            X (dict): Dictionary of features.

        Returns:
            np.ndarray: Predicted probabilities for the positive class.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()
        loader = create_dataloader(
            X, batch_size=self.params["batch_size"] * 2, shuffle=False, num_workers=0
        )

        all_probs = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = self._to_tensor_dict(batch_X)
                logits = self.model(batch_X)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())

        return np.array(all_probs)
