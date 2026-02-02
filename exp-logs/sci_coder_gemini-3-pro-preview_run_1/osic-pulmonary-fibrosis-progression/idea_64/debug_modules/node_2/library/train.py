import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.model import TSBCNet, laplace_log_likelihood_loss
from library.data import LungDataset, get_transforms
from library.utils import calculate_metric


class Trainer:
    """
    Handles the training, validation, and inference loops for the TSBC-Net model.
    """

    def __init__(self, model, optimizer, scheduler, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.best_score = -float("inf")
        self.best_model_path = os.path.join(
            self.config.CHECKPOINT_DIR, "best_model.pth"
        )

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            img_ax = batch["image_axial"].to(self.device)
            img_cor = batch["image_coronal"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            target = batch["target"].to(self.device)

            # Meta info for trajectory calculation
            week_num = torch.tensor(batch["meta"]["Week_Num"]).float().to(self.device)
            base_fvc = (
                torch.tensor(batch["meta"]["Baseline_FVC"]).float().to(self.device)
            )
            base_week = (
                torch.tensor(batch["meta"]["Baseline_Week"]).float().to(self.device)
            )

            self.optimizer.zero_grad()

            # Forward pass -> Parameters [alpha, sigma_base, sigma_growth]
            preds = self.model(img_ax, img_cor, tabular)
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Calculate Trajectory
            # FVC = Baseline + alpha * (Current_Week - Baseline_Week)
            dt = week_num - base_week
            fvc_pred = base_fvc + alpha * dt

            # Sigma = Base + Growth * |dt|
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            # Loss
            loss = laplace_log_likelihood_loss(target, fvc_pred, sigma_pred)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * img_ax.size(0)

        return running_loss / len(loader.dataset)

    def validate(self, loader):
        self.model.eval()
        all_true = []
        all_pred = []
        all_sigma = []

        with torch.no_grad():
            for batch in loader:
                img_ax = batch["image_axial"].to(self.device)
                img_cor = batch["image_coronal"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                target = batch["target"].float().numpy()

                week_num = (
                    torch.tensor(batch["meta"]["Week_Num"]).float().to(self.device)
                )
                base_fvc = (
                    torch.tensor(batch["meta"]["Baseline_FVC"]).float().to(self.device)
                )
                base_week = (
                    torch.tensor(batch["meta"]["Baseline_Week"]).float().to(self.device)
                )

                # Forward
                preds = self.model(img_ax, img_cor, tabular)
                alpha = preds[:, 0]
                sigma_base = preds[:, 1]
                sigma_growth = preds[:, 2]

                # Trajectory
                dt = week_num - base_week
                fvc_pred = base_fvc + alpha * dt
                sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

                all_true.extend(target)
                all_pred.extend(fvc_pred.cpu().numpy())
                all_sigma.extend(sigma_pred.cpu().numpy())

        score = calculate_metric(
            np.array(all_true), np.array(all_pred), np.array(all_sigma)
        )
        return score

    def fit(self, train_loader, val_loader, epochs, patience):
        print("Starting Training...")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_score = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
            )

            if val_score > self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New Best Model Saved! Score: {self.best_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader, output_path):
        print(f"Loading best model from {self.best_model_path} for inference...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        results = []

        with torch.no_grad():
            for batch in test_loader:
                img_ax = batch["image_axial"].to(self.device)
                img_cor = batch["image_coronal"].to(self.device)
                tabular = batch["tabular"].to(self.device)

                # Meta
                patients = batch["meta"]["Patient"]
                week_num = (
                    torch.tensor(batch["meta"]["Week_Num"]).float().to(self.device)
                )
                base_fvc = (
                    torch.tensor(batch["meta"]["Baseline_FVC"]).float().to(self.device)
                )
                base_week = (
                    torch.tensor(batch["meta"]["Baseline_Week"]).float().to(self.device)
                )

                # Forward
                preds = self.model(img_ax, img_cor, tabular)
                alpha = preds[:, 0]
                sigma_base = preds[:, 1]
                sigma_growth = preds[:, 2]

                # Trajectory
                dt = week_num - base_week
                fvc_pred = base_fvc + alpha * dt
                sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

                # Collect results
                fvc_np = fvc_pred.cpu().numpy()
                sigma_np = sigma_pred.cpu().numpy()
                week_np = week_num.cpu().numpy().astype(int)

                for i in range(len(patients)):
                    pid = patients[i]
                    wk = week_np[i]
                    fvc = fvc_np[i]
                    conf = sigma_np[i]

                    # Clip confidence for submission as per metric requirement
                    conf = max(conf, 70.0)

                    results.append(
                        {"Patient_Week": f"{pid}_{wk}", "FVC": fvc, "Confidence": conf}
                    )

        # Save submission
        sub_df = pd.DataFrame(results)
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run(debug=False, epochs=None, batch_size=None):
    """
    Main execution function to setup data, model, and run training.

    Args:
        debug (bool): If True, runs on a subset of data.
        epochs (int): Number of training epochs. Defaults to Config.EPOCHS.
        batch_size (int): Batch size. Defaults to Config.BATCH_SIZE.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = Config.DEVICE

    if epochs is None:
        epochs = Config.EPOCHS
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    print(f"Using device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        # Test needs to be processed to ensure pipeline works, but we can subset
        test_df = test_df.head(Config.DEBUG_SAMPLES)

    # 3. Datasets and Loaders
    # Note: LungDataset handles image caching internally
    train_ds = LungDataset(train_df, mode="train", transform=get_transforms("train"))
    val_ds = LungDataset(val_df, mode="val", transform=get_transforms("val"))
    test_ds = LungDataset(test_df, mode="test", transform=get_transforms("test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Model Setup
    model = TSBCNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Initialize Trainer and Run
    trainer = Trainer(model, optimizer, scheduler, device, Config)

    # Fit model
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=Config.PATIENCE)

    # Generate Submission
    # Saving to ./submission/submission.csv as requested
    submission_path = "./submission/submission.csv"
    trainer.predict(test_loader, output_path=submission_path)
