import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import MetricMonitor, calculate_f1


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        classes=None,
        save_dir="./working/idea_2",
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            device (torch.device): The device to use (CPU or GPU).
            classes (np.ndarray): Array of class labels.
            save_dir (str): Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.classes = classes
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Batch-level scheduler step (e.g., OneCycleLR)
            if self.scheduler and isinstance(
                self.scheduler, optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

            metric_monitor.update("Loss", loss.item())

        print(f"Epoch {epoch} Train | {metric_monitor}")

    def validate(self, val_loader):
        """
        Validates the model on the validation set.
        """
        self.model.eval()
        metric_monitor = MetricMonitor()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, targets)
                metric_monitor.update("Loss", loss.item())

                preds = torch.argmax(output, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        f1 = calculate_f1(np.array(all_targets), np.array(all_preds))
        print(f"Validation | {metric_monitor} | F1: {f1}")
        return f1

    def fit(
        self, train_loader, val_loader, epochs, patience=5, checkpoint_name="model.pth"
    ):
        """
        Runs the training loop with early stopping.
        """
        best_f1 = -float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, checkpoint_name)

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            self.train_one_epoch(train_loader, epoch)
            val_f1 = self.validate(val_loader)

            # Epoch-level scheduler step
            if self.scheduler and not isinstance(
                self.scheduler, optim.lr_scheduler.OneCycleLR
            ):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                if self.classes is not None:
                    # Cite debug_lesson_2: Couple Model Configuration Explicitly with Checkpoints
                    torch.save(
                        {
                            "state_dict": self.model.state_dict(),
                            "classes": self.classes,
                        },
                        best_model_path,
                    )
                else:
                    torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with F1: {best_f1}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model weights
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

    def predict(self, test_loader, idx2cat, output_file="./submission/submission.csv"):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.model.eval()
        predictions = []
        image_ids = []

        print("Generating predictions...")
        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)
                output = self.model(images)
                preds = torch.argmax(output, dim=1).cpu().numpy()

                image_ids.extend(ids.numpy())
                predictions.extend(preds)

        # Map predictions to category_ids
        predicted_cats = [idx2cat[p] for p in predictions]

        # Create submission dataframe
        df = pd.DataFrame({"Id": image_ids, "Predicted": predicted_cats})

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
