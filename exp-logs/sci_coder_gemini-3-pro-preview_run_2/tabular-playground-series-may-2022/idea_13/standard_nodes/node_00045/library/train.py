import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import set_seed, get_device, compute_auc
from library.data import get_dataloaders
from library.model import HybridResFunnel


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        save_path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.best_auc = 0.0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for x_cont, x_cat, y in self.train_loader:
            x_cont = x_cont.to(self.device)
            x_cat = x_cat.to(self.device)
            y = y.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Model outputs probabilities (sigmoid already applied)
            preds = self.model(x_cont, x_cat)

            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_cont, x_cat, y in self.val_loader:
                x_cont = x_cont.to(self.device)
                x_cat = x_cat.to(self.device)

                preds = self.model(x_cont, x_cat)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        auc = compute_auc(all_targets, all_preds)
        return auc

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_auc = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} | LR: {current_lr:.1e} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            # Early Stopping / Checkpointing based on AUC
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved with AUC: {self.best_auc:.10f}")

        print(f"Training complete. Best Validation AUC: {self.best_auc:.10f}")


def predict(model, test_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, _ in test_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            preds = self_preds = model(x_cont, x_cat)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training(
    epochs=35,
    batch_size=1024,
    learning_rate=1e-3,
    weight_decay=1e-2,
    work_dir="./working/idea_13",
    input_dir="./input",
    metadata_dir="./metadata",
):
    # 1. Setup
    set_seed(42)
    device = get_device()
    os.makedirs(work_dir, exist_ok=True)
    model_save_path = os.path.join(work_dir, "best_model.pth")
    submission_path = os.path.join(work_dir, "submission.csv")

    print(f"Device: {device}")
    print(f"Working Directory: {work_dir}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=4,
        load_cached_data=True,
        cache_dir=work_dir,
        input_dir=input_dir,
        metadata_dir=metadata_dir,
    )

    # 3. Model Initialization
    print("Initializing Hybrid ResFunnel model...")
    model = HybridResFunnel(
        num_continuous=30,
        vocab_size=32,  # A-Z mapped to 1-26, plus padding
        embedding_dim=32,
        seq_len=10,
        transformer_layers=2,
        backbone_dims=[512, 256, 128],
        dropout=0.35,
    )
    model.to(device)

    # 4. Optimization
    # AdamW with high weight decay as specified
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # StepLR: decay by 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # BCELoss because model outputs sigmoid probabilities
    criterion = nn.BCELoss()

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_path=model_save_path,
    )

    trainer.fit(epochs=epochs)

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # 7. Submission
    print("Creating submission file...")
    # Read test metadata to get IDs
    test_meta_path = os.path.join(metadata_dir, "test_metadata.csv")
    if os.path.exists(test_meta_path):
        df_test_meta = pd.read_csv(test_meta_path)
        test_ids = df_test_meta["id"].values
    else:
        # Fallback if metadata missing (unlikely given setup)
        print("Warning: Test metadata not found. Reading from raw test.csv...")
        df_test_raw = pd.read_csv(os.path.join(input_dir, "test.csv"))
        test_ids = df_test_raw["id"].values

    if len(test_ids) != len(predictions):
        print(
            f"Error: Mismatch between test IDs ({len(test_ids)}) and predictions ({len(predictions)})"
        )

    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
