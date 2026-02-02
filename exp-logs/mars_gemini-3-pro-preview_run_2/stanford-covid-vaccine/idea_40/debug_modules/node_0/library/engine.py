import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.config import Config
from library.loss import MCRMSELoss
from library.model import TAFRDNModel
from library.utils import set_seed, mcrmse


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        set_seed(config.seed)

        # Initialize Model
        self.model = TAFRDNModel(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        # Loss Function
        # We score reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        self.criterion = MCRMSELoss(
            scored_indices=config.scored_cols, seq_scored=config.pred_len
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for inputs, partner_indices, targets in train_loader:
            inputs = inputs.to(self.device)
            partner_indices = partner_indices.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass returns list of outputs [pass1_preds, pass2_preds]
            outputs = self.model(inputs, partner_indices)

            # Calculate loss
            # L_total = L_pass2 + 0.5 * L_pass1
            # Note: outputs[-1] is the final pass (Pass 2), outputs[-2] is Pass 1

            final_pred = outputs[-1]
            loss_final = self.criterion(final_pred, targets)

            total_loss = loss_final

            # Add auxiliary loss if we have multiple passes
            if len(outputs) > 1:
                aux_pred = outputs[-2]  # Pass 1
                loss_aux = self.criterion(aux_pred, targets)
                total_loss = total_loss + self.config.aux_loss_weight * loss_aux

            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            self.optimizer.step()

            running_loss += total_loss.item()

        return running_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        running_score = 0.0

        # Determine indices for scoring manually for metric logging if needed,
        # but the criterion handles it. We will use the criterion value as the metric.

        with torch.no_grad():
            for inputs, partner_indices, targets in val_loader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs, partner_indices)
                final_pred = outputs[-1]

                # Metric calculation
                score = self.criterion(final_pred, targets)
                running_score += score.item()

        return running_score / len(val_loader)

    def train(self, train_loader, val_loader):
        print(f"Starting training on {self.device}...")
        best_val_score = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.evaluate(val_loader)

            # Print full precision
            print(
                f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss} - Val MCRMSE: {val_score}"
            )

            self.scheduler.step(val_score)

            # Checkpoint & Early Stopping
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.best_model_path)
                # print(f"New best model saved to {self.config.best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val MCRMSE: {best_val_score}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Loading best model for inference...")
        if not os.path.exists(self.config.best_model_path):
            print(
                "No best model found. Using current model state (warning: untrained?)."
            )
        else:
            self.model.load_state_dict(
                torch.load(self.config.best_model_path, map_location=self.device)
            )

        self.model.eval()

        all_ids = []
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs, partner_indices, _ in test_loader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)

                # We don't have targets in test, but loader yields 3 items
                ids = test_loader.dataset.ids[len(all_ids) : len(all_ids) + len(inputs)]

                outputs = self.model(inputs, partner_indices)
                final_pred = outputs[-1]  # (B, SeqLen, 5)

                all_preds.append(final_pred.cpu().numpy())
                all_ids.extend(ids)

        # Concatenate all batches
        # Shape: (TotalSamples, SeqLen, 5)
        predictions = np.concatenate(all_preds, axis=0)

        # Prepare submission DataFrame
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_data = []

        target_cols = (
            self.config.target_cols
        )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        for i, sample_id in enumerate(all_ids):
            # predictions[i] shape is (107, 5)
            sample_pred = predictions[i]

            for seq_pos in range(self.config.seq_len):
                row_id = f"{sample_id}_{seq_pos}"
                row_values = sample_pred[seq_pos]

                # Create dict for this row
                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_values[col_idx])

                submission_data.append(row_dict)

        submission_df = pd.DataFrame(submission_data)

        # Save
        submission_df.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")
