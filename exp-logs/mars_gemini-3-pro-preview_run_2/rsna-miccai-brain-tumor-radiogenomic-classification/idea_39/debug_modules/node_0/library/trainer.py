import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from library import config, utils, data_loader, model


class Trainer:
    """
    Manages the training, validation, and checkpointing of a single model.
    """

    def __init__(self, model, train_loader, val_loader, device, save_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_name = save_name
        self.logger = utils.get_logger(f"TRAINER_{save_name}")

        # Optimization setup
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # State tracking
        self.best_auc = 0.0
        self.patience = 5  # Early stopping patience
        self.counter = 0
        self.best_model_path = os.path.join(
            config.MODEL_SAVE_DIR, f"{self.save_name}_best.pth"
        )

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            # BCEWithLogitsLoss expects target shape (B, 1) matching output
            target = target.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

            # Store predictions for monitoring
            preds = torch.sigmoid(output).detach().cpu().numpy()
            targets = target.detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

        epoch_loss = running_loss / len(self.train_loader.dataset)

        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device).unsqueeze(1)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item() * data.size(0)

                preds = torch.sigmoid(output).cpu().numpy()
                targets = target.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)

        val_loss = running_loss / len(self.val_loader.dataset)

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self, epochs=config.NUM_EPOCHS):
        self.logger.info(f"Starting training for {self.save_name}...")

        for epoch in range(1, epochs + 1):
            train_loss, train_auc = self.train_epoch(epoch)
            val_loss, val_auc = self.validate()

            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.10f}, Train AUC: {train_auc:.10f} - "
                f"Val Loss: {val_loss:.10f}, Val AUC: {val_auc:.10f}"
            )

            # Checkpointing
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"New best model saved to {self.best_model_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        return self.best_model_path


def predict_with_tta(model_instance, loader, device):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Averages predictions from:
    1. Original
    2. Horizontal Flip (dim 3)
    3. Vertical Flip (dim 2)
    """
    model_instance.eval()
    all_preds = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)

            # 1. Original
            out_orig = model_instance(data)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (Width is dim 3 in B,C,H,W)
            data_h = torch.flip(data, dims=[3])
            out_h = model_instance(data_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (Height is dim 2 in B,C,H,W)
            data_v = torch.flip(data, dims=[2])
            out_v = model_instance(data_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            all_preds.extend(avg_prob.cpu().numpy().flatten())

    return np.array(all_preds)


def run_training_phase(train_ds, val_ds, save_name):
    """
    Sets up and runs the training process for a specific expert model.
    """
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    net = model.AsymmetricEfficientNet()

    # Initialize Trainer
    trainer = Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        save_name=save_name,
    )

    # Run Training
    best_model_path = trainer.fit()
    return best_model_path


def run_experiment():
    """
    Orchestrates the full Dual-Scale Ensemble pipeline:
    1. Load Data
    2. Train Texture Expert (Stride 2)
    3. Train Context Expert (Stride 5)
    4. Generate Predictions (with TTA)
    5. Ensemble and Save Submission
    """
    logger = utils.get_logger("EXPERIMENT")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Loading Datasets...")
    train_ds_A, train_ds_B, val_ds_A, val_ds_B = data_loader.get_datasets()

    # -------------------------------------------------------------------------
    # 2. Train Texture Expert (Model A)
    # -------------------------------------------------------------------------
    logger.info(">>> Starting Phase A: Texture Expert (Stride 2)")
    path_A = run_training_phase(train_ds_A, val_ds_A, "model_A_texture")

    # -------------------------------------------------------------------------
    # 3. Train Context Expert (Model B)
    # -------------------------------------------------------------------------
    logger.info(">>> Starting Phase B: Context Expert (Stride 5)")
    path_B = run_training_phase(train_ds_B, val_ds_B, "model_B_context")

    # -------------------------------------------------------------------------
    # 4. Inference & Ensemble
    # -------------------------------------------------------------------------
    logger.info(">>> Starting Inference Phase")

    # Load Test Data
    test_ds_A, test_ds_B = data_loader.get_test_datasets()
    test_ids = data_loader.get_test_ids()

    test_loader_A = DataLoader(
        test_ds_A,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader_B = DataLoader(
        test_ds_B,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Load Best Model A
    net_A = model.AsymmetricEfficientNet().to(config.DEVICE)
    net_A.load_state_dict(torch.load(path_A, map_location=config.DEVICE))

    # Predict A
    logger.info("Predicting with Texture Expert...")
    preds_A = predict_with_tta(net_A, test_loader_A, config.DEVICE)

    # Load Best Model B
    net_B = model.AsymmetricEfficientNet().to(config.DEVICE)
    net_B.load_state_dict(torch.load(path_B, map_location=config.DEVICE))

    # Predict B
    logger.info("Predicting with Context Expert...")
    preds_B = predict_with_tta(net_B, test_loader_B, config.DEVICE)

    # Ensemble Average
    final_preds = (preds_A + preds_B) / 2.0

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    logger.info("Generating Submission File...")

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": final_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_PATH}")
