import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from library import config
from library import dataset
from library import model as model_lib


class Trainer:
    """
    Trainer class to handle the training and validation loops.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device, save_path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.best_acc = -1.0

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs, patience):
        """
        Runs the full training process with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        early_stopping_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # Check for improvement
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Validation accuracy improved. Model saved to {self.save_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"No improvement. Early stopping counter: {early_stopping_counter}/{patience}"
                )

            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model weights before returning
        if os.path.exists(self.save_path):
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )

        return self.model


def train_model(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    patience=5,
):
    """
    Main function to setup data, model, and run training.
    """
    # Ensure reproducibility
    config.set_seed(config.SEED)

    # Load Metadata
    train_csv = os.path.join(config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(config.METADATA_DIR, "val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(
            "Metadata CSV files not found. Please run metadata generation first."
        )

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # Create Datasets
    # Phase='train' triggers balancing (undersampling unknown, oversampling silence)
    # Apply SpecAugment during training (Cite solution_lesson_node_00003)
    spec_augment = utils.SpecAugment()
    train_dataset = dataset.SpeechCommandsDataset(
        df_train, phase="train", transform=spec_augment
    )
    val_dataset = dataset.SpeechCommandsDataset(df_val, phase="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    device = torch.device(config.DEVICE)
    model = model_lib.ResNetAudioClassifier(num_classes=config.NUM_CLASSES).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define save path for the best model
    save_dir = config.WORK_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    # Instantiate Trainer and Fit
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, save_path
    )
    best_model = trainer.fit(epochs=epochs, patience=patience)

    return best_model


def generate_submission(model, batch_size=config.BATCH_SIZE):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    print("Generating submission...")
    device = torch.device(config.DEVICE)
    model.eval()
    model.to(device)

    # Load Test Metadata
    test_csv = os.path.join(config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError("Test metadata CSV not found.")

    df_test = pd.read_csv(test_csv)

    # Create Dataset and Loader
    test_dataset = dataset.SpeechCommandsDataset(df_test, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = []
    filenames = df_test["filepath"].apply(os.path.basename).tolist()

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Apply Softmax to get probabilities (optional for argmax but good practice)
            probs = torch.softmax(outputs, dim=1)

            # Get predicted class indices
            _, preds = torch.max(probs, 1)
            predictions.extend(preds.cpu().numpy())

    # Map indices back to string labels
    pred_labels = [config.IDX_TO_LABEL[idx] for idx in predictions]

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"fname": filenames, "label": pred_labels})

    # Save to CSV
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def run_experiment():
    """
    Orchestrates the full pipeline: training and submission generation.
    """
    model = train_model()
    generate_submission(model)
