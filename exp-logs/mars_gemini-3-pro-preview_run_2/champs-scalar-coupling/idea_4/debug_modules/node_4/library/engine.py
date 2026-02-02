import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import TrainConfig, ModelConfig
from library.data import MoleculeDataset, collate_graphs, TYPE_MAP
from library.model import HybridModel

# Inverse mapping to convert integer type indices back to strings (e.g., 0 -> '1JHC')
INV_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: Computation device.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move batch data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(batch)
        targets = batch["coupling_target"]

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Step scheduler if it is updated per epoch
    if scheduler is not None:
        scheduler.step()

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device, norm_stats):
    """
    Evaluates the model on the validation set using LogMAE.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: Computation device.
        norm_stats: Dictionary of normalization statistics (mean, std) per coupling type.

    Returns:
        float: The mean of the Log MAE across all coupling types.
    """
    model.eval()

    # Storage for predictions and targets per coupling type
    results = {t: {"preds": [], "targets": []} for t in TYPE_MAP.keys()}

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            targets = batch["coupling_target"]
            types = batch["coupling_type"]

            # Move to CPU for numpy processing
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
            types = types.cpu().numpy()

            for i in range(len(preds)):
                t_idx = types[i]
                t_str = INV_TYPE_MAP[t_idx]

                p = preds[i]
                t = targets[i]

                # Un-normalize predictions and targets to original scale
                if t_str in norm_stats:
                    stats = norm_stats[t_str]
                    if stats["std"] > 1e-7:
                        p = p * stats["std"] + stats["mean"]
                        t = t * stats["std"] + stats["mean"]

                results[t_str]["preds"].append(p)
                results[t_str]["targets"].append(t)

    # Calculate Log MAE per type
    log_maes = []
    print("\nValidation Metrics per Type:")
    for t_str, data in results.items():
        if not data["preds"]:
            continue

        p_arr = np.array(data["preds"])
        t_arr = np.array(data["targets"])

        # Mean Absolute Error
        mae = np.mean(np.abs(p_arr - t_arr))

        # Log of MAE (using natural log as is standard unless specified otherwise)
        # Avoid log(0)
        log_mae = np.log(mae + 1e-9)
        log_maes.append(log_mae)

        print(f"  Type {t_str}: MAE={mae:.6f}, LogMAE={log_mae:.6f}")

    final_score = np.mean(log_maes) if log_maes else 0.0
    return final_score


def predict(model, loader, device, norm_stats, output_path=None):
    """
    Generates predictions for a dataset (usually test set).

    Args:
        model: The PyTorch model.
        loader: DataLoader for the dataset.
        device: Computation device.
        norm_stats: Normalization stats from training set to un-normalize predictions.
        output_path: Optional path to save the predictions as CSV.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'scalar_coupling_constant'.
    """
    model.eval()
    ids = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            types = batch["coupling_type"]
            batch_ids = batch["coupling_id"]

            preds = preds.cpu().numpy()
            types = types.cpu().numpy()
            batch_ids = batch_ids.cpu().numpy()

            for i in range(len(preds)):
                t_idx = types[i]
                t_str = INV_TYPE_MAP[t_idx]

                p = preds[i]

                # Un-normalize
                if t_str in norm_stats:
                    stats = norm_stats[t_str]
                    if stats["std"] > 1e-7:
                        p = p * stats["std"] + stats["mean"]

                ids.append(batch_ids[i])
                predictions.append(p)

    df = pd.DataFrame({"id": ids, "scalar_coupling_constant": predictions})

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


class Engine:
    """
    Encapsulates the training, validation, and prediction lifecycle.
    """

    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        self.config = train_config
        self.model_config = model_config
        self.device = torch.device(self.config.device)

        set_seed(42)

        # Initialize Datasets
        print("Initializing Datasets...")
        self.train_dataset = MoleculeDataset(
            split="train", config=self.config, model_config=self.model_config
        )
        self.val_dataset = MoleculeDataset(
            split="val", config=self.config, model_config=self.model_config
        )

        # Store normalization stats for use in validation/inference
        self.norm_stats = self.train_dataset.norm_stats

        # Initialize Loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
            pin_memory=True,
        )

        # Initialize Model
        print("Initializing Model...")
        self.model = HybridModel(self.model_config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Loss Function (MAE on normalized targets)
        self.criterion = nn.L1Loss()

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=1e-6
        )

    def train(self):
        """
        Runs the training loop with early stopping.
        """
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device} for {self.config.epochs} epochs.")

        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.scheduler,
            )

            # Validate
            val_score = validate(
                self.model, self.val_loader, self.device, self.norm_stats
            )

            print(
                f"Epoch {epoch}/{self.config.epochs} | Train Loss: {train_loss:.6f} | Val LogMAE: {val_score:.6f}"
            )

            # Early Stopping and Checkpointing
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.config.model_path)
                print(f"  New best model saved to {self.config.model_path}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{self.config.patience}"
                )

            if patience_counter >= self.config.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val LogMAE: {best_score:.6f}")

    def generate_submission(self):
        """
        Loads the best model and generates predictions for the test set.
        """
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(self.config.model_path):
            self.model.load_state_dict(
                torch.load(self.config.model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found, using current model state.")

        # Initialize Test Dataset
        test_dataset = MoleculeDataset(
            split="test", config=self.config, model_config=self.model_config
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
        )

        # Predict and Save
        predict(
            self.model,
            test_loader,
            self.device,
            self.norm_stats,
            output_path=self.config.submission_path,
        )
        print(f"Submission saved to {self.config.submission_path}")
