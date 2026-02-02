import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.utils import set_seed, AverageMeter, calculate_macro_f1
from library.dataset import create_dataloaders
from library.model import PlantClassifier


class Trainer:
    """
    Trainer class for Plant Classification using EfficientNetV2.
    Handles training with Mixup, validation, and inference with TTA.
    """

    def __init__(self, config, id2label=None):
        self.config = config
        self.id2label = id2label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.working_dir = "./working/idea_2"
        os.makedirs(self.working_dir, exist_ok=True)

        # Initialize Model
        # Using 15501 classes as per dataset analysis
        self.model = PlantClassifier(
            num_classes=15501, model_name="tf_efficientnetv2_s", pretrained=True
        )
        self.model.to(self.device)

        # Loss Function with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-2),
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        self.best_f1 = 0.0

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        losses = AverageMeter()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        macro_f1 = calculate_macro_f1(all_labels, all_preds)
        return losses.avg, macro_f1

    def fit(self, train_loader, val_loader, epochs=15):
        """
        Runs the training loop with Mixup and Early Stopping.
        """
        # Scheduler setup
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.get("lr", 1e-3),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=100.0,
        )

        patience = self.config.get("patience", 3)
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Training Phase
            self.model.train()
            train_losses = AverageMeter()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                # Apply Mixup
                use_mixup = self.config.get("use_mixup", True) and batch_size > 1

                if use_mixup:
                    alpha = 1.0
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(batch_size).to(self.device)

                    mixed_images = lam * images + (1 - lam) * images[index]
                    y_a, y_b = labels, labels[index]

                    with autocast():
                        outputs = self.model(mixed_images)
                        loss = lam * self.criterion(outputs, y_a) + (
                            1 - lam
                        ) * self.criterion(outputs, y_b)
                else:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()

                train_losses.update(loss.item(), batch_size)

            # Validation Phase
            val_loss, val_f1 = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} | Time: {epoch_time}s | "
                f"Train Loss: {train_losses.avg} | "
                f"Val Loss: {val_loss} | "
                f"Val F1: {val_f1}"
            )

            # Checkpointing
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                save_path = os.path.join(self.working_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set using TTA (Horizontal Flip).
        Saves the result to submission.csv.
        """
        # Load best model
        model_path = os.path.join(self.working_dir, "best_model.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded best model from {model_path}")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        predictions = []
        image_ids = []

        print("Generating predictions with TTA (Horizontal Flip)...")

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)

                with autocast():
                    # Original image
                    output1 = self.model(images)
                    # Horizontally flipped image
                    output2 = self.model(torch.flip(images, dims=[3]))

                    # Average logits
                    avg_output = (output1 + output2) / 2

                preds = torch.argmax(avg_output, dim=1).cpu().numpy()

                predictions.extend(preds)
                image_ids.extend(ids)

        # Map predictions back to original category_ids
        if self.id2label:
            predictions = [self.id2label[p] for p in predictions]

        # Prepare submission dataframe
        # Id in sample submission is int, we receive strings from dataset
        submission_df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

        # Ensure Id is integer for sorting
        submission_df["Id"] = submission_df["Id"].astype(int)
        submission_df = submission_df.sort_values("Id").reset_index(drop=True)

        # Save
        os.makedirs("submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


def run_training(debug=False, epochs=15, batch_size=64):
    """
    Main entry point to run the training and prediction pipeline.
    """
    set_seed(42)

    # Create DataLoaders
    # Using 260 as defined in dataset.py, batch_size 64 fits well on A100 for EffNetV2-S
    train_loader, val_loader, test_loader = create_dataloaders(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        debug=debug,
        img_size=260,
    )

    config = {"lr": 1e-3, "weight_decay": 1e-2, "patience": 5, "use_mixup": True}

    trainer = Trainer(config)

    # Train
    trainer.fit(train_loader, val_loader, epochs=epochs)

    # Predict
    trainer.predict(test_loader)
