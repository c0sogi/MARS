import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder

from library.config import Config, ModelEMA, set_seed, load_or_create_metadata
from library.dataset import SpeechCommandDataset, MixupCollate
from library.model import DilatedEfficientNet
from library.utils import map_fine_grained_to_12_class


class Trainer:
    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.label_encoder = LabelEncoder()

    def setup_data(self):
        # Load metadata with caching
        df_all = load_or_create_metadata(load_cached_data=True)

        # Split into train and validation
        df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
        df_val = df_all[df_all["split"] == "val"].reset_index(drop=True)

        # Fit Encoder on all available fine labels to ensure coverage
        self.label_encoder.fit(df_all["fine_label"])
        self.num_classes = len(self.label_encoder.classes_)

        # Weighted Sampler for Training to handle class imbalance
        class_counts = df_train["fine_label"].value_counts()
        weights = []
        for label in df_train["fine_label"]:
            # Inverse frequency weighting
            count = class_counts.get(label, 1)
            weights.append(1.0 / count)

        sampler = WeightedRandomSampler(
            weights, num_samples=len(df_train), replacement=True
        )

        # Initialize Datasets
        train_ds = SpeechCommandDataset(df_train, self.label_encoder, is_train=True)
        val_ds = SpeechCommandDataset(df_val, self.label_encoder, is_train=False)

        # Initialize DataLoaders
        # Train loader uses MixupCollate for augmentation
        self.train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=MixupCollate(alpha=Config.MIXUP_ALPHA),
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print(f"Data Setup Complete. Classes: {self.num_classes}")
        print(f"Train Size: {len(df_train)}, Val Size: {len(df_val)}")

    def train(self):
        set_seed(Config.SEED)
        self.setup_data()

        # Model Setup
        model = DilatedEfficientNet(self.num_classes).to(self.device)
        ema = ModelEMA(model, Config.EMA_DECAY)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = -1.0
        patience = 10
        patience_counter = 0

        print("Starting Training...")

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0

            for batch_idx, (images, targets_a, targets_b, lam) in enumerate(
                self.train_loader
            ):
                images = images.to(self.device)
                targets_a = targets_a.to(self.device)
                targets_b = targets_b.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)

                # Mixup Loss Calculation
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )

                loss.backward()
                optimizer.step()

                # Update EMA model
                ema.update(model)

                train_loss += loss.item()

            scheduler.step()
            avg_train_loss = train_loss / len(self.train_loader)

            # Validation using EMA model
            val_loss, val_acc = self.validate(ema.ema, criterion)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Save Best Model & Early Stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                os.makedirs(Config.CACHE_DIR, exist_ok=True)
                torch.save(
                    ema.ema.state_dict(),
                    os.path.join(Config.CACHE_DIR, "best_model.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Best Validation Accuracy: {best_acc}")

    def validate(self, model, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / len(self.val_loader), correct / total

    def generate_submission(self):
        set_seed(Config.SEED)

        # Ensure encoder is fitted (if running separately)
        if not hasattr(self.label_encoder, "classes_"):
            # Reload metadata to fit encoder
            df_all = load_or_create_metadata(load_cached_data=True)
            self.label_encoder.fit(df_all["fine_label"])
            self.num_classes = len(self.label_encoder.classes_)

        # Load Test Data
        df_test = pd.read_csv(Config.TEST_METADATA)
        # Placeholder fine_label for dataset compatibility
        df_test["fine_label"] = Config.UNKNOWN_LABEL

        test_ds = SpeechCommandDataset(df_test, self.label_encoder, is_train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Load Best Model
        model = DilatedEfficientNet(self.num_classes).to(self.device)
        model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return

        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted_indices = torch.max(outputs, 1)

                predicted_labels = self.label_encoder.inverse_transform(
                    predicted_indices.cpu().numpy()
                )
                predictions.extend(predicted_labels)

        # Map fine-grained labels to 12-class format
        final_labels = [map_fine_grained_to_12_class(label) for label in predictions]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "fname": df_test["filepath"].apply(os.path.basename),
                "label": final_labels,
            }
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
