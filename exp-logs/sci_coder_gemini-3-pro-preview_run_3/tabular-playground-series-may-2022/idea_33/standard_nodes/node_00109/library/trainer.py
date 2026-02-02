import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.utils import seed_everything, get_device
from library.dataset import get_datasets
from library.model import CRHPEModel


class Trainer:
    """
    Encapsulates the training, validation, and inference logic for the CR-HPE model.
    """

    def __init__(self, model, device, optimizer, scheduler, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for cat_x, cont_x, y in train_loader:
            cat_x = cat_x.to(self.device)
            cont_x = cont_x.to(self.device)
            y = y.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward pass returns list of outputs from 5 streams
            outputs = self.model(cat_x, cont_x)

            # Sum loss across all streams
            loss = 0
            for out in outputs:
                loss += self.criterion(out, y)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for cat_x, cont_x, y in val_loader:
                cat_x = cat_x.to(self.device)
                cont_x = cont_x.to(self.device)

                outputs = self.model(cat_x, cont_x)

                # Ensemble averaging (Sigmoid -> Mean)
                probs = torch.zeros_like(outputs[0])
                for out in outputs:
                    probs += torch.sigmoid(out)
                probs /= len(outputs)

                preds.append(probs.cpu().numpy())
                targets.append(y.numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        return roc_auc_score(targets, preds)

    def predict(self, test_loader):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for cat_x, cont_x in test_loader:
                cat_x = cat_x.to(self.device)
                cont_x = cont_x.to(self.device)

                outputs = self.model(cat_x, cont_x)

                # Ensemble averaging (Sigmoid -> Mean)
                probs = torch.zeros_like(outputs[0])
                for out in outputs:
                    probs += torch.sigmoid(out)
                probs /= len(outputs)

                preds.append(probs.cpu().numpy())

        return np.concatenate(preds).flatten()


def run_training(
    epochs=50,
    batch_size=1024,
    load_cached_data=True,
    base_dir="./metadata",
    cache_dir="./working/idea_33",
    debug=False,
):
    """
    Main function to execute the training pipeline.
    """
    seed_everything(42)
    device = get_device()

    print(f"Initializing training on {device}...")

    # 1. Load Datasets
    # get_datasets handles caching and processing internally via library.data_processing
    train_dataset, val_dataset, test_dataset, vocab_sizes, test_ids = get_datasets(
        load_cached_data=load_cached_data,
        base_dir=base_dir,
        cache_dir=cache_dir,
        debug=debug,
    )

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Initialize Model
    # Determine continuous feature count from the dataset tensor shape
    num_cont = train_dataset.cont_features.shape[1]
    model = CRHPEModel(vocab_sizes, num_cont).to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    # 5. Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 6. Initialize Trainer
    trainer = Trainer(model, device, optimizer, scheduler, criterion)

    best_auc = 0.0
    best_model_path = "./working/best_model.pth"

    # 7. Training Loop
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_auc = trainer.validate(val_loader)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Val AUC: {val_auc:.15f}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation AUC: {best_auc:.15f}")

    # 8. Inference and Submission
    print("Generating predictions on Test Set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_preds = trainer.predict(test_loader)

    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"id": test_ids, "target": test_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
