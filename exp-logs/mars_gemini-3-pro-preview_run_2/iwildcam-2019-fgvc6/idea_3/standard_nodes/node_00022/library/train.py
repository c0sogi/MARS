import os
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library import utils, data, model


class Trainer:
    def __init__(self, debug=Config.DEBUG):
        self.device = Config.DEVICE
        self.debug = debug

        # Initialize DataLoaders
        self.train_loader, self.val_loader, self.test_loader = data.get_dataloaders(
            debug=self.debug
        )

        # Initialize Model
        self.model = model.EfficientNetClassifier().to(self.device)

        # Loss Function with Class Weights
        train_metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)
        class_weights = utils.compute_class_weights(train_metadata)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)

        # Best Model State
        self.best_model_state = None
        self.best_f1 = 0.0

    def train_one_epoch(self, optimizer):
        self.model.train()
        running_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        val_f1 = utils.calculate_macro_f1(all_labels, all_preds)

        return val_loss, val_f1

    def run_stage1(self):
        print("\n=== Starting Stage 1: Head Alignment (Frozen Backbone) ===")

        # Freeze Backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Optimizer for Head only
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=Config.LR_STAGE1)

        for epoch in range(Config.EPOCHS_STAGE1):
            train_loss = self.train_one_epoch(optimizer)
            val_loss, val_f1 = self.validate()

            print(
                f"Stage 1 | Epoch {epoch+1}/{Config.EPOCHS_STAGE1} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F1: {val_f1}"
            )

            # Save best model even in Stage 1
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_model_state = copy.deepcopy(self.model.state_dict())

    def run_stage2(self):
        print("\n=== Starting Stage 2: Fine-Tuning (Unfrozen Top Blocks) ===")

        # Unfreeze the last few blocks of the backbone
        # EfficientNet features are a Sequential container.
        # We unfreeze the last 3 blocks (arbitrary choice based on 'top blocks' strategy)
        # Total blocks in B3 features is 9.

        # First, ensure everything is frozen (sanity check)
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 3 blocks
        blocks = list(self.model.backbone.children())
        for block in blocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also ensure the head is trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Optimizer (AdamW)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=Config.LR_STAGE2,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.SCHEDULER_T_MAX
        )

        patience_counter = 0

        for epoch in range(Config.EPOCHS_STAGE2):
            train_loss = self.train_one_epoch(optimizer)
            val_loss, val_f1 = self.validate()

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Stage 2 | Epoch {epoch+1}/{Config.EPOCHS_STAGE2} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F1: {val_f1}"
            )

            if val_f1 > self.best_f1:
                print(f"New best F1! ({self.best_f1} -> {val_f1}) Saving model...")
                self.best_f1 = val_f1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save to disk immediately
                torch.save(self.best_model_state, Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

    def generate_submission(self):
        print("\n=== Generating Submission ===")

        # Load best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights.")
        else:
            print("Warning: No best model state found. Using current weights.")

        self.model.eval()
        ids = []
        predictions = []

        with torch.no_grad():
            for images, image_ids in self.test_loader:
                images = images.to(self.device)

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                ids.extend(image_ids)
                predictions.extend(preds)

        # Create DataFrame
        submission_df = pd.DataFrame({"Id": ids, "Category": predictions})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())

    def fit(self):
        utils.set_seed()
        self.run_stage1()
        self.run_stage2()
        self.generate_submission()


def run_training():
    """
    Main entry point for training.
    """
    trainer = Trainer()
    trainer.fit()
