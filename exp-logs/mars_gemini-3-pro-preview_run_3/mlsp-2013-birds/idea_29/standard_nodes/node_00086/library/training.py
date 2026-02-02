import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, MixupHandler
from library.data import get_dataloaders
from library.models import get_cnn_model, SymbolicMLP


class SnapshotManager:
    """
    Manages saving and deleting checkpoints to maintain the Top-K best models
    based on validation AUC.
    """

    def __init__(
        self, save_dir, model_name, fold_idx, num_snapshots=Config.NUM_SNAPSHOTS
    ):
        self.save_dir = save_dir
        self.model_name = model_name
        self.fold_idx = fold_idx
        self.num_snapshots = num_snapshots
        # List of tuples: (score, epoch, filepath)
        self.snapshots = []

    def update(self, score, epoch, model):
        """
        Updates the snapshot list with the current model state if the score qualifies.
        """
        filename = f"{self.model_name}_fold{self.fold_idx}_ep{epoch}_auc{score:.6f}.pth"
        filepath = os.path.join(self.save_dir, filename)

        # Add new candidate
        self.snapshots.append((score, epoch, filepath))

        # Sort by score descending
        self.snapshots.sort(key=lambda x: x[0], reverse=True)

        # Check if current model should be saved (is in top K)
        should_save = False
        for s in self.snapshots[: self.num_snapshots]:
            if s[2] == filepath:
                should_save = True
                break

        if should_save:
            torch.save(model.state_dict(), filepath)

            # Remove models that fell out of top K
            while len(self.snapshots) > self.num_snapshots:
                _, _, path_to_remove = self.snapshots.pop()
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
        else:
            # If not in top K, just remove from list (it wasn't saved)
            self.snapshots = [s for s in self.snapshots if s[2] != filepath]

    def get_best_score(self):
        if not self.snapshots:
            return 0.0
        return self.snapshots[0][0]


class Trainer:
    """
    Handles the training and validation loop for a single model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        model_name,
        fold_idx,
        is_mlp=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.fold_idx = fold_idx
        self.is_mlp = is_mlp

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

        # Determine save directory based on model type
        if self.is_mlp:
            self.save_dir = os.path.join(Config.CHECKPOINT_DIR, "mlp")
        else:
            self.save_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)

        self.snapshot_manager = SnapshotManager(
            self.save_dir, self.model_name, self.fold_idx
        )

        self.mixup = MixupHandler(alpha=Config.MIXUP_ALPHA)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in self.train_loader:
            # Extract data based on stream type
            if self.is_mlp:
                inputs = batch["features"].to(self.device)
            else:
                inputs = batch["image"].to(self.device)

            targets = batch["labels"].to(self.device)

            # Apply Mixup
            inputs, targets = self.mixup.apply(inputs, targets)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                if self.is_mlp:
                    inputs = batch["features"].to(self.device)
                else:
                    inputs = batch["image"].to(self.device)

                targets = batch["labels"].to(self.device)

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.cpu())
                all_targets.append(targets.cpu())

        if not all_preds:
            return 0.0

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        score = calculate_roc_auc(all_targets, all_preds)
        return score

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        best_score = 0.0
        patience_counter = 0

        print(f"Starting training for {self.model_name} - Fold {self.fold_idx}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            # Step scheduler
            self.scheduler.step()

            # Print metrics
            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val AUC = {val_score:.16f}"
            )

            # Update snapshots
            self.snapshot_manager.update(val_score, epoch, self.model)

            # Early Stopping Check
            # We track the absolute best score for patience purposes
            current_best = self.snapshot_manager.get_best_score()
            if current_best > best_score:
                best_score = current_best
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Score: {best_score:.16f}"
                )
                break

        print(
            f"Finished training {self.model_name} - Fold {self.fold_idx}. Best AUC: {best_score:.16f}"
        )


def run_training(debug_epochs=None, load_cached_data=True):
    """
    Orchestrates the training for all models across all folds.

    Args:
        debug_epochs (int, optional): If set, overrides Config.EPOCHS for debugging.
        load_cached_data (bool): Whether to use cached data splits/features.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = debug_epochs if debug_epochs is not None else Config.EPOCHS

    # Iterate over folds
    for fold_idx in range(Config.N_FOLDS):
        print(f"\n{'='*20} Processing Fold {fold_idx} {'='*20}")

        # Get DataLoaders for this fold
        dataloaders = get_dataloaders(fold_idx, load_cached_data=load_cached_data)

        # ---------------------------------------------------------------------
        # Stream A: Deep Learning (CNNs)
        # ---------------------------------------------------------------------
        for model_name in Config.CNN_MODELS:
            print(f"\n--- Training CNN: {model_name} ---")

            # Initialize Model
            model = get_cnn_model(model_name, pretrained=True).to(device)

            # Initialize Trainer
            trainer = Trainer(
                model=model,
                train_loader=dataloaders["train_cnn"],
                val_loader=dataloaders["val_cnn"],
                device=device,
                model_name=model_name,
                fold_idx=fold_idx,
                is_mlp=False,
            )

            # Train
            trainer.fit(epochs=epochs)

            # Clean up to save memory
            del model, trainer
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Stream B: Symbolic Learning (MLP)
        # ---------------------------------------------------------------------
        print(f"\n--- Training Symbolic MLP ---")

        # Initialize Model
        mlp_model = SymbolicMLP().to(device)

        # Initialize Trainer
        trainer = Trainer(
            model=mlp_model,
            train_loader=dataloaders["train_mlp"],
            val_loader=dataloaders["val_mlp"],
            device=device,
            model_name="mlp",
            fold_idx=fold_idx,
            is_mlp=True,
        )

        # Train
        trainer.fit(epochs=epochs)

        # Clean up
        del mlp_model, trainer
        torch.cuda.empty_cache()

    print("\nAll training tasks completed.")
