import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.loss import MaskedMSELoss
from library.data import get_dataloaders
from library.model import RNAModel


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )
        self.criterion = MaskedMSELoss()

        self.best_mcrmse = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            seq = batch["seq"].to(self.device)
            loop = batch["loop"].to(self.device)
            dist = batch["dist"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(seq, loop, dist)
            loss = self.criterion(preds, targets)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / n_batches if n_batches > 0 else 0.0

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                seq = batch["seq"].to(self.device)
                loop = batch["loop"].to(self.device)
                dist = batch["dist"].to(self.device)
                targets = batch["targets"].to(self.device)

                preds = self.model(seq, loop, dist)

                # Collect data for metric calculation
                # We must slice to the scored length (68) for accurate MCRMSE calculation
                # as the metric is defined only on scored positions.
                preds_sliced = preds[:, : Config.PRED_LEN, :]
                targets_sliced = targets[:, : Config.PRED_LEN, :]

                all_preds.append(preds_sliced.cpu().numpy())
                all_targets.append(targets_sliced.cpu().numpy())

        if not all_preds:
            return 0.0

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute MCRMSE
        score = compute_mcrmse(all_preds, all_targets)
        return score

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_mcrmse = self.validate()

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_mcrmse}"
            )

            # Checkpoint
            if val_mcrmse < self.best_mcrmse:
                print(
                    f"Validation score improved ({self.best_mcrmse} -> {val_mcrmse}). Saving model..."
                )
                self.best_mcrmse = val_mcrmse
                torch.save(self.model.state_dict(), Config.MODEL_PATH)

        print(f"Training complete. Best Val MCRMSE: {self.best_mcrmse}")

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found, using current model state.")

        self.model.eval()

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in self.test_loader:
                seq = batch["seq"].to(self.device)
                loop = batch["loop"].to(self.device)
                dist = batch["dist"].to(self.device)
                batch_ids = batch["id"]

                # Predict (Batch, 107, 3)
                preds = self.model(seq, loop, dist)

                preds_np = preds.cpu().numpy()

                ids_list.extend(batch_ids)
                preds_list.append(preds_np)

        # Concatenate all predictions: (N_samples, 107, 3)
        all_preds = np.concatenate(preds_list, axis=0)

        # Prepare data for DataFrame
        # We need to flatten: N_samples * 107 rows
        # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        submission_data = []

        # The model predicts: [reactivity, deg_Mg_pH10, deg_Mg_50C]
        # We need to insert 0.0 for [deg_pH10, deg_50C]

        for idx, sample_id in enumerate(ids_list):
            sample_preds = all_preds[idx]  # (107, 3)

            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"

                # Extract predicted values
                reactivity = float(sample_preds[seqpos, 0])
                deg_Mg_pH10 = float(sample_preds[seqpos, 1])
                deg_Mg_50C = float(sample_preds[seqpos, 2])

                # Unscored/Unpredicted columns set to 0
                deg_pH10 = 0.0
                deg_50C = 0.0

                submission_data.append(
                    {
                        "id_seqpos": row_id,
                        "reactivity": reactivity,
                        "deg_Mg_pH10": deg_Mg_pH10,
                        "deg_pH10": deg_pH10,
                        "deg_Mg_50C": deg_Mg_50C,
                        "deg_50C": deg_50C,
                    }
                )

        # Create DataFrame
        df_sub = pd.DataFrame(submission_data)

        # Ensure column order matches sample submission
        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        df_sub = df_sub[cols]

        # Save
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training(debug_size=None):
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug_size=debug_size
    )

    # 3. Model
    model = RNAModel().to(device)

    # 4. Trainer
    trainer = Trainer(model, device, train_loader, val_loader, test_loader)

    # 5. Execute
    trainer.fit()
    trainer.generate_submission()


if __name__ == "__main__":
    # Can be run directly for testing, though typically imported
    run_training()
