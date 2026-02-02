import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.metrics import calculate_final_score


class Engine:
    """
    Handles the training and validation loops for the Toxicity Model.
    Implements Device-Side Trimming and Multi-Task Loss calculation.
    """

    def __init__(self, model, optimizer, scheduler, device):
        """
        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer.
            scheduler: The learning rate scheduler.
            device: The device (CPU/GPU) to run on.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Binary Cross Entropy with Logits for multi-label/binary classification
        self.criterion = nn.BCEWithLogitsLoss()

    def _trim_tensors(self, input_ids, attention_mask):
        """
        Performs Device-Side Trimming.
        Slices the input tensors to the maximum effective length in the current batch.
        """
        # Calculate the maximum length of non-padding tokens in this batch
        # attention_mask is 1 for tokens, 0 for padding.
        # sum(dim=1) gives the length of each sequence.
        # max() gives the longest sequence in the batch.
        max_len = int(attention_mask.sum(dim=1).max().item())

        # Slice the tensors to reduce computation on padding
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        return input_ids, attention_mask

    def train_one_epoch(self, data_loader, epoch_index):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            # 1. Move to Device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            # 2. Device-Side Trimming
            input_ids, attention_mask = self._trim_tensors(input_ids, attention_mask)

            # 3. Zero Gradients
            self.optimizer.zero_grad()

            # 4. Forward Pass
            # Model outputs: (toxicity_logits, aux_logits)
            toxicity_logits, aux_logits = self.model(input_ids, attention_mask)

            # 5. Loss Calculation
            # targets shape: (Batch, 1 + Num_Aux)
            # Column 0 is the main toxicity target
            main_targets = targets[:, 0].unsqueeze(1)
            # Columns 1: end are the auxiliary identity targets
            aux_targets = targets[:, 1:]

            loss_main = self.criterion(toxicity_logits, main_targets)
            loss_aux = self.criterion(aux_logits, aux_targets)

            # Weighted sum
            loss = loss_main + (Config.AUX_LOSS_WEIGHT * loss_aux)

            # 6. Backward Pass
            loss.backward()

            # 7. Optimization
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch_index} Training Loss: {avg_loss}")
        return avg_loss

    def validate(self, data_loader):
        """
        Evaluates the model on the validation set.
        Computes the competition metric (Weighted ROC-AUCs).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Lists to store predictions and targets for metric calculation
        all_preds = []
        all_targets = []
        all_ids = []

        with torch.no_grad():
            for batch in data_loader:
                # 1. Move to Device
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                targets = batch["targets"].to(self.device, non_blocking=True)
                ids = batch["ids"].numpy()  # Keep IDs on CPU for dataframe construction

                # 2. Device-Side Trimming
                input_ids, attention_mask = self._trim_tensors(
                    input_ids, attention_mask
                )

                # 3. Forward Pass
                toxicity_logits, aux_logits = self.model(input_ids, attention_mask)

                # 4. Loss Calculation (for monitoring)
                main_targets = targets[:, 0].unsqueeze(1)
                aux_targets = targets[:, 1:]

                loss_main = self.criterion(toxicity_logits, main_targets)
                loss_aux = self.criterion(aux_logits, aux_targets)
                loss = loss_main + (Config.AUX_LOSS_WEIGHT * loss_aux)

                total_loss += loss.item()
                num_batches += 1

                # 5. Store Predictions
                # Apply sigmoid to get probabilities [0, 1]
                preds = torch.sigmoid(toxicity_logits).detach().cpu().numpy()

                # We need the full targets (including identities) for the bias metrics
                # targets shape is (B, 1 + Num_Aux)
                target_vals = targets.detach().cpu().numpy()

                all_preds.append(preds)
                all_targets.append(target_vals)
                all_ids.append(ids)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0).flatten()  # Shape (N,)
        all_targets = np.concatenate(all_targets, axis=0)  # Shape (N, 1 + Num_Aux)
        all_ids = np.concatenate(all_ids, axis=0)

        # Reconstruct DataFrame for Metric Calculation
        # We need columns: id, target, prediction, and all identity columns

        # 1. Create base dictionary
        data_dict = {
            "id": all_ids,
            "target": all_targets[:, 0],  # Main toxicity target
            "prediction": all_preds,
        }

        # 2. Add identity columns
        # Config.IDENTITY_COLUMNS matches the order in targets[:, 1:]
        for i, identity_col in enumerate(Config.IDENTITY_COLUMNS):
            data_dict[identity_col] = all_targets[:, i + 1]

        val_df = pd.DataFrame(data_dict)

        # Calculate Competition Metrics
        final_score, detailed_results = calculate_final_score(
            val_df, prediction_col="prediction", target_col="target"
        )

        print(f"Validation Loss: {avg_loss}")
        print(f"Validation Score: {final_score}")
        print(f"Detailed Results: {detailed_results}")

        return avg_loss, final_score
