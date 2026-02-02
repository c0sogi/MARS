import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
from library.config import Config
from library.utils import seed_everything, mcrmse, average_weights
from library.loss import MaskedMSELoss
from library.model import RNAModel


class SWAHandler:
    """
    Manages the collection and averaging of model checkpoints for Stochastic Weight Averaging (SWA).
    """

    def __init__(self, start_epoch, save_dir):
        self.start_epoch = start_epoch
        self.save_dir = save_dir
        self.checkpoints = []
        os.makedirs(save_dir, exist_ok=True)

    def update(self, model, epoch):
        """
        Collects the model state if the current epoch is within the SWA window.
        States are moved to CPU to preserve GPU memory.
        """
        if epoch >= self.start_epoch:
            state_dict = copy.deepcopy(model.state_dict())
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            self.checkpoints.append(cpu_state_dict)

    def finalize(self):
        """
        Computes the arithmetic mean of the collected checkpoints.
        Returns the averaged state_dict.
        """
        if not self.checkpoints:
            print("SWA: No checkpoints collected. Returning None.")
            return None

        print(f"SWA: Averaging {len(self.checkpoints)} checkpoints...")
        avg_state = average_weights(self.checkpoints)
        return avg_state


class Engine:
    """
    Handles training, evaluation, and prediction loops.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = MaskedMSELoss()

    def train_one_epoch(self, dataloader, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch in dataloader:
            seq = batch["seq"].to(self.device)
            loop = batch["loop"].to(self.device)
            dist = batch["dist"].to(self.device)
            targets = batch["targets"].to(self.device)

            batch_size = seq.size(0)
            dataset_size += batch_size

            self.optimizer.zero_grad()

            outputs = self.model(seq, loop, dist)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += loss.item() * batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns loss and MCRMSE score.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                seq = batch["seq"].to(self.device)
                loop = batch["loop"].to(self.device)
                dist = batch["dist"].to(self.device)
                targets = batch["targets"].to(self.device)

                batch_size = seq.size(0)
                dataset_size += batch_size

                outputs = self.model(seq, loop, dist)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * batch_size

                # Slice to scored length for metric calculation
                # outputs: (B, 107, 3) -> (B, 68, 3)
                pred_scored = outputs[:, : Config.PRED_LEN, :].cpu().numpy()
                target_scored = targets[:, : Config.PRED_LEN, :].cpu().numpy()

                all_preds.append(pred_scored)
                all_targets.append(target_scored)

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE
        score = mcrmse(all_targets, all_preds)

        return epoch_loss, score

    def predict(self, dataloader):
        """
        Generates predictions for the test set.
        Returns sample IDs and predictions array of shape (N, 107, 3).
        """
        self.model.eval()
        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in dataloader:
                seq = batch["seq"].to(self.device)
                loop = batch["loop"].to(self.device)
                dist = batch["dist"].to(self.device)
                ids = batch["id"]

                # Predict full length (107)
                outputs = self.model(seq, loop, dist)

                all_preds.append(outputs.cpu().numpy())
                all_ids.extend(ids)

        return all_ids, np.concatenate(all_preds, axis=0)


def run_training(train_loader, val_loader, test_loader):
    """
    Main function to run the training pipeline, SWA, and submission generation.
    """
    # ensure directories exist
    _ = Config()

    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    engine = Engine(model, device, optimizer, scheduler)
    swa_handler = SWAHandler(Config.SWA_START_EPOCH, Config.WORKING_DIR)

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = engine.train_one_epoch(train_loader, epoch)
        val_loss, val_score = engine.evaluate(val_loader)

        # Step scheduler at the end of epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.2e} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCRMSE: {val_score}"
        )

        # Update SWA handler
        swa_handler.update(model, epoch)

    # Finalize SWA
    print("Finalizing SWA Model...")
    avg_state = swa_handler.finalize()

    if avg_state is not None:
        model.load_state_dict(avg_state)
        # Re-evaluate with SWA model
        val_loss, val_score = engine.evaluate(val_loader)
        print(f"SWA Model | Val Loss: {val_loss} | Val MCRMSE: {val_score}")

        # Save SWA model
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"Saved SWA model to {Config.MODEL_SAVE_PATH}")
    else:
        print("SWA failed, using last model state.")
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # Generate Submission
    print("Generating submission...")
    ids, preds = engine.predict(test_loader)

    # preds shape: (N, 107, 3)
    # Target columns: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Submission columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []

    # Map prediction channel indices to submission column names
    # Channel 0 -> reactivity
    # Channel 1 -> deg_Mg_pH10
    # Channel 2 -> deg_Mg_50C

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            reactivity = float(sample_preds[seqpos, 0])
            deg_Mg_pH10 = float(sample_preds[seqpos, 1])
            deg_Mg_50C = float(sample_preds[seqpos, 2])

            # Unscored columns set to 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    submission_df = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
