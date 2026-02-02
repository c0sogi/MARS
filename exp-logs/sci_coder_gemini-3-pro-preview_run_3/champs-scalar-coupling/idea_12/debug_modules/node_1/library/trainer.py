import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, TargetScaler, MetricLogger
from library.layers import get_model
from library.loss import PhysicsInformedLoss


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Scalar Coupling Prediction model.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else Config()
        self.device = self.config.DEVICE

        # Set seeds for reproducibility
        seed_everything(self.config.SEED)

        # Components
        self.model = get_model(self.config).to(self.device)
        self.criterion = PhysicsInformedLoss(self.config)
        self.scaler = TargetScaler()

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

    def fit_scaler(self, data_dir):
        """
        Fits the TargetScaler using the pre-processed training data arrays.
        """
        print("Fitting TargetScaler...")
        vals_path = os.path.join(data_dir, "train_coupling_values.npy")
        types_path = os.path.join(data_dir, "train_coupling_types.npy")

        if not os.path.exists(vals_path) or not os.path.exists(types_path):
            raise FileNotFoundError(
                "Processed training data not found. Run data processing first."
            )

        train_vals = np.load(vals_path)
        train_types = np.load(types_path)

        # Manually populate scaler statistics
        for i, t_str in enumerate(self.config.COUPLING_TYPES):
            mask = train_types == i
            if np.any(mask):
                vals = train_vals[mask]
                self.scaler.mean_arr[i] = np.mean(vals)
                self.scaler.std_arr[i] = np.std(vals)
                self.scaler.means[t_str] = float(np.mean(vals))
                self.scaler.stds[t_str] = float(np.std(vals))
            else:
                self.scaler.mean_arr[i] = 0.0
                self.scaler.std_arr[i] = 1.0

        self.scaler.fitted = True
        print("TargetScaler fitted.")

    def train(self, train_loader, val_loader, epochs=None):
        """
        Executes the training loop with validation and early stopping.
        """
        if not self.scaler.fitted:
            # Attempt to fit scaler from the loader's dataset directory
            self.fit_scaler(train_loader.dataset.data_dir)

        max_epochs = epochs if epochs is not None else self.config.MAX_EPOCHS
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(max_epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss_accum = 0.0
            num_batches = 0

            for batch in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                out = self.model(
                    batch["atom_types"],
                    batch["edge_index"],
                    batch["edge_dist"],
                    batch["triplet_index"],
                    batch["triplet_angle"],
                    batch["coupling_node_indices"],
                    batch["coupling_edge_indices"],
                    batch["coupling_types"],
                )

                # Prepare Targets
                # Standardize coupling values
                means = torch.tensor(self.scaler.mean_arr, device=self.device)[
                    batch["coupling_types"]
                ]
                stds = torch.tensor(self.scaler.std_arr, device=self.device)[
                    batch["coupling_types"]
                ]
                target_coupling = (batch["coupling_values"] - means) / stds

                targets = {
                    "coupling": target_coupling,
                    "shielding": batch["aux_shielding"],
                    "charge": batch["aux_charges"],
                }

                # Compute Loss
                loss, _ = self.criterion(out, targets)

                loss.backward()
                self.optimizer.step()

                train_loss_accum += loss.item()
                num_batches += 1

            avg_train_loss = train_loss_accum / max(1, num_batches)
            self.scheduler.step()

            # --- Validation Phase ---
            self.model.eval()
            logger = MetricLogger()

            with torch.no_grad():
                for batch in val_loader:
                    out = self.model(
                        batch["atom_types"],
                        batch["edge_index"],
                        batch["edge_dist"],
                        batch["triplet_index"],
                        batch["triplet_angle"],
                        batch["coupling_node_indices"],
                        batch["coupling_edge_indices"],
                        batch["coupling_types"],
                    )

                    # Standardize validation targets for metric logging
                    means = torch.tensor(self.scaler.mean_arr, device=self.device)[
                        batch["coupling_types"]
                    ]
                    stds = torch.tensor(self.scaler.std_arr, device=self.device)[
                        batch["coupling_types"]
                    ]
                    target_coupling = (batch["coupling_values"] - means) / stds

                    # Update logger with scaled predictions and targets
                    logger.update(
                        out["coupling"].squeeze(),
                        target_coupling,
                        batch["coupling_types"],
                    )

            # Compute LMAE (MetricLogger handles inverse transform)
            val_score = logger.compute_metric(self.scaler)

            print(
                f"Epoch {epoch+1}: Train Loss {avg_train_loss:.6f}, Val LMAE {val_score:.9f}"
            )

            # --- Checkpointing & Early Stopping ---
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Val LMAE: {best_score:.9f}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        Returns predictions in the original physical scale.
        """
        print("Loading best model for inference...")
        if not os.path.exists(self.config.MODEL_SAVE_PATH):
            raise FileNotFoundError("No saved model found. Train the model first.")

        self.model.load_state_dict(
            torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        if not self.scaler.fitted:
            self.fit_scaler(test_loader.dataset.data_dir)

        preds = []

        with torch.no_grad():
            for batch in test_loader:
                out = self.model(
                    batch["atom_types"],
                    batch["edge_index"],
                    batch["edge_dist"],
                    batch["triplet_index"],
                    batch["triplet_angle"],
                    batch["coupling_node_indices"],
                    batch["coupling_edge_indices"],
                    batch["coupling_types"],
                )

                # Inverse transform immediately
                batch_preds = self.scaler.inverse_transform(
                    out["coupling"].squeeze(), batch["coupling_types"]
                )
                preds.append(batch_preds)

        return np.concatenate(preds)

    def generate_submission(self, test_loader):
        """
        Generates predictions and saves the submission CSV file.
        Aligns predictions with the correct IDs from metadata.
        """
        # 1. Get Predictions
        raw_preds = self.predict(test_loader)

        # 2. Align with IDs
        # The loader iterates molecules in sorted order of molecule_name.
        # We must load the test metadata and sort it identically to extract IDs.
        print("Aligning predictions with test IDs...")
        test_meta = pd.read_csv(self.config.TEST_META_PATH)

        # Group by molecule and sort groups by name
        test_grp = test_meta.groupby("molecule_name")
        sorted_mols = sorted(test_grp.groups.keys())

        ids_ordered = []
        for m in sorted_mols:
            # Extract IDs for this molecule in the order they appear (which matches processing order)
            ids_ordered.append(test_grp.get_group(m)["id"].values)

        ids_ordered = np.concatenate(ids_ordered)

        if len(ids_ordered) != len(raw_preds):
            raise ValueError(
                f"Mismatch: {len(ids_ordered)} IDs vs {len(raw_preds)} predictions."
            )

        # 3. Create Submission DataFrame
        # Map ID to Prediction
        pred_map = dict(zip(ids_ordered, raw_preds))

        # Load sample submission to ensure correct row order
        sub_df = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)
        sub_df["scalar_coupling_constant"] = sub_df["id"].map(pred_map)

        # 4. Save
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
