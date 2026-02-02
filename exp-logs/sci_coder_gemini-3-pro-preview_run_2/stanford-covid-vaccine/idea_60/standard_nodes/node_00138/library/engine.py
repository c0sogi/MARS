import torch
import numpy as np
import pandas as pd
import os
import time
from library.config import Config
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import AHIRN


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Engine:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = AHIRN().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = MCRMSELoss()

        # Identify indices of the targets that are actually scored in the competition
        self.scored_indices = [
            i for i, t in enumerate(Config.ALL_TARGETS) if t in Config.SCORED_TARGETS
        ]

    def train_epoch(self, loader):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            partners = batch["partner_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass returns predictions from both passes
            y1, y2 = self.model(inputs, partners)

            # Calculate loss for both passes
            loss1 = self.criterion(y1, targets)
            loss2 = self.criterion(y2, targets)

            # Weighted sum: L_total = L_pass2 + 0.5 * L_pass1
            loss = loss2 + Config.AUX_LOSS_WEIGHT * loss1

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(loader.dataset)

    def validate(self, loader):
        """
        Runs validation.
        Calculates:
        1. Average Loss (consistent with training objective).
        2. Global MCRMSE on Scored Targets (competition metric).
        """
        self.model.eval()

        # Accumulators for Global MCRMSE (Scored Columns)
        # We accumulate Sum of Squared Errors (SSE) and Counts globally
        total_sse = torch.zeros(len(self.scored_indices), device=self.device)
        total_count = 0

        running_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                partners = batch["partner_indices"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Inference uses pass 2 (refined predictions)
                _, y2 = self.model(inputs, partners)

                # 1. Standard Loss (All columns, Batch-averaged)
                loss = self.criterion(y2, targets)
                running_loss += loss.item() * inputs.size(0)

                # 2. Competition Metric (Scored columns, Global accumulation)
                # Slice to scored length (Cite debug_lesson_8)
                preds_scored = y2[:, : Config.SEQ_SCORED, self.scored_indices]
                targets_scored = targets[:, : Config.SEQ_SCORED, self.scored_indices]

                # Squared Error: (B, L_scored, n_scored)
                se = (preds_scored - targets_scored) ** 2

                # Sum over Batch and Sequence dimensions
                batch_sse = se.sum(dim=(0, 1))
                total_sse += batch_sse

                # Total elements per column in this batch
                total_count += inputs.size(0) * Config.SEQ_SCORED

        # Compute Global RMSE per column: sqrt(Sum_SSE / Total_Count)
        rmse_per_col = torch.sqrt(total_sse / total_count)

        # MCRMSE is the mean of the column-wise RMSEs
        global_mcrmse = rmse_per_col.mean().item()

        avg_loss = running_loss / len(loader.dataset)

        return avg_loss, global_mcrmse

    def predict(self, loader):
        """Runs inference on the test set."""
        self.model.eval()
        preds_list = []
        ids_list = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                partners = batch["partner_indices"].to(self.device)
                ids = batch["id"]

                _, y2 = self.model(inputs, partners)

                preds_list.append(y2.cpu().numpy())
                ids_list.extend(ids)

        return np.concatenate(preds_list, axis=0), ids_list

    def run(self):
        """Main execution flow."""
        set_seed(Config.SEED)

        # Ensure directories exist
        os.makedirs(Config.IDEA_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        print("Loading data...")
        # Load data with caching enabled
        train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

        print(f"Starting training on {self.device}...")
        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_metric = self.validate(val_loader)

            # Step scheduler based on validation loss
            self.scheduler.step(val_loss)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Global MCRMSE (Scored): {val_metric:.10f}"
            )

            # Early Stopping Check
            # We use val_loss (all targets) for model selection to ensure general stability
            if val_loss < best_metric:
                best_metric = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
                print(f"  New best model saved! Loss: {best_metric:.6f}")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # --- Inference ---
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )

        test_preds, test_ids = self.predict(test_loader)

        print("Generating submission file...")
        submission_data = []

        for i, sample_id in enumerate(test_ids):
            # pred_matrix: (107, 5)
            pred_matrix = test_preds[i]

            for pos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{pos}"
                row_vals = pred_matrix[pos].tolist()
                submission_data.append([row_id] + row_vals)

        columns = ["id_seqpos"] + Config.ALL_TARGETS
        sub_df = pd.DataFrame(submission_data, columns=columns)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_engine():
    engine = Engine()
    engine.run()
