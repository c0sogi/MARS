import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.loss import MCRMSELoss
from library.model import SR_DCN


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RNAEngine:
    """
    Engine class to handle training, validation, and submission generation
    for the Stabilized Recurrent Dense-Context Network (SR-DCN).
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed(Config.SEED)

        # Identify indices of columns used for scoring
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

    def train_fn(self, model, dataloader, optimizer, criterion, aux_weight=0.5):
        """
        Performs one epoch of training using the Stabilized Recurrent strategy.
        """
        model.train()
        running_loss = 0.0
        dataset_size = 0

        for inputs, partner_indices, targets, _ in dataloader:
            inputs = inputs.to(self.device)
            partner_indices = partner_indices.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()

            # --- Pass 1: Cold Start ---
            # Recycling channels initialized to zeros inside the model if None is passed
            preds_1 = model(inputs, partner_indices, recycling=None)

            # --- Pass 2: Refinement ---
            # Detach predictions to stop gradient flow back to Pass 1 through the recycling input
            # This stabilizes the feedback loop
            recycling_input = preds_1.detach()
            preds_2 = model(inputs, partner_indices, recycling=recycling_input)

            # --- Loss Calculation ---
            # Calculate loss for both passes
            loss_2 = criterion(preds_2, targets)  # Primary Loss (Refined)
            loss_1 = criterion(preds_1, targets)  # Auxiliary Loss (Hint)

            # Combined loss
            total_loss = loss_2 + (aux_weight * loss_1)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    def eval_fn(self, model, dataloader, criterion):
        """
        Evaluates the model and computes the Global MCRMSE.
        Accumulates SSE across the whole dataset to avoid batch-averaging bias.
        """
        model.eval()

        # Accumulators for Global MCRMSE
        # We need sum of squared errors per scored column
        num_scored = len(self.scored_indices)
        sum_squared_errors = torch.zeros(num_scored, device=self.device)
        total_scored_elements = 0

        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for inputs, partner_indices, targets, _ in dataloader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.size(0)

                # --- Pass 1 ---
                preds_1 = model(inputs, partner_indices, recycling=None)

                # --- Pass 2 ---
                # Use Pass 1 output as recycling input
                preds_2 = model(inputs, partner_indices, recycling=preds_1)

                # Calculate batch loss for logging purposes
                batch_loss = criterion(preds_2, targets)
                running_loss += batch_loss.item() * batch_size
                dataset_size += batch_size

                # --- Global Metric Accumulation ---
                # 1. Slice to scored sequence length (68)
                preds_sliced = preds_2[:, : Config.SCORED_LEN, :]
                targets_sliced = targets[:, : Config.SCORED_LEN, :]

                # 2. Select only the scored columns
                preds_scored = preds_sliced[:, :, self.scored_indices]
                targets_scored = targets_sliced[:, :, self.scored_indices]

                # 3. Compute Squared Errors
                diff = preds_scored - targets_scored
                squared_diff = diff**2

                # Sum over Batch (0) and Sequence (1) dimensions -> Shape: (NumScoredCols,)
                batch_sse = squared_diff.sum(dim=(0, 1))

                sum_squared_errors += batch_sse
                total_scored_elements += batch_size * Config.SCORED_LEN

        # Compute Global RMSE per column
        # MSE = SSE / N
        mse_per_col = sum_squared_errors / total_scored_elements
        rmse_per_col = torch.sqrt(mse_per_col)

        # Compute MCRMSE (mean of RMSEs across columns)
        global_mcrmse = torch.mean(rmse_per_col).item()
        avg_loss = running_loss / dataset_size

        return global_mcrmse, avg_loss

    def run_training(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        model = SR_DCN().to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.FACTOR,
            patience=Config.PATIENCE,
        )
        criterion = MCRMSELoss().to(self.device)

        best_score = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_fn(
                model,
                train_loader,
                optimizer,
                criterion,
                aux_weight=Config.AUX_LOSS_WEIGHT,
            )
            val_score, val_loss = self.eval_fn(model, val_loader, criterion)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val MCRMSE: {val_score}"
            )

            scheduler.step(val_score)

            # Early Stopping Check
            if val_score < best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val MCRMSE: {best_score}")
        return best_score

    def generate_submission(self, test_loader, submission_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        model = SR_DCN().to(self.device)
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        if not os.path.exists(best_model_path):
            print(f"Error: Best model not found at {best_model_path}")
            return

        print(f"Loading model from {best_model_path} for inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for inputs, partner_indices, _, ids in test_loader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)

                # Inference with recycling (Pass 1 -> Pass 2)
                preds_1 = model(inputs, partner_indices, recycling=None)
                preds_2 = model(inputs, partner_indices, recycling=preds_1)

                # Move to CPU
                preds_np = preds_2.cpu().numpy()  # Shape: (B, 107, 5)

                all_ids.extend(ids)
                all_preds.append(preds_np)

        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N_samples, 107, 5)

        # Flatten for submission
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_data = []
        target_cols = Config.TARGET_COLS  # Ensures correct column order

        for i, sample_id in enumerate(all_ids):
            sample_preds = all_preds[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                # Construct row ID
                row_id = f"{sample_id}_{seqpos}"

                # Get predictions for this position
                row_values = sample_preds[seqpos]

                # Create row list
                row = [row_id] + row_values.tolist()
                submission_data.append(row)

        columns = ["id_seqpos"] + target_cols
        df_sub = pd.DataFrame(submission_data, columns=columns)

        # Ensure directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
