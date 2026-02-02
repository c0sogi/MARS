import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.loss_fn import WeightedMultiTaskLoss
from library.metrics import BiasMetricCalculator


class Engine:
    """
    Engine class to handle training, evaluation, and prediction loops.
    Encapsulates the 'Pad-then-Trim' logic and SWA updates.
    """

    def __init__(self, model, optimizer, device, scheduler=None, swa_handler=None):
        """
        Args:
            model (nn.Module): The PyTorch model.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (torch.device): The device to run on.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            swa_handler (SWAHandler, optional): Handler for Stochastic Weight Averaging.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.swa_handler = swa_handler
        self.criterion = WeightedMultiTaskLoss()

    def _trim_batch(self, input_ids, attention_mask):
        """
        Trims input tensors to the maximum sequence length in the batch.
        This implements the 'Pad-then-Trim' strategy for efficiency.
        """
        # Find the maximum length where attention_mask is 1
        # attention_mask shape: (batch, seq_len)
        active_lens = attention_mask.sum(dim=1)
        max_len = active_lens.max().item()

        # Slice the tensors
        # Ensure we don't slice below a minimum length (e.g. 1) to avoid errors
        max_len = max(max_len, 1)

        return input_ids[:, :max_len], attention_mask[:, :max_len]

    def train_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): The training data loader.
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # 1. Move to device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            aux_targets = batch["aux_targets"].to(self.device, non_blocking=True)
            weights = batch["weight"].to(self.device, non_blocking=True)

            # 2. Trim tensors
            input_ids, attention_mask = self._trim_batch(input_ids, attention_mask)

            # 3. Zero Gradients
            self.optimizer.zero_grad()

            # 4. Forward Pass
            # Expects tuple: (toxicity_logits, identity_logits)
            tox_logits, ident_logits = self.model(input_ids, attention_mask)

            # 5. Compute Loss
            loss = self.criterion(
                tox_logits, ident_logits, targets, aux_targets, weights
            )

            # 6. Backward Pass
            loss.backward()

            # 7. Clip Gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_GRAD_NORM)

            # 8. Optimizer Step
            self.optimizer.step()

            # 9. Scheduler Step (OneCycleLR steps per batch)
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        # 10. SWA Update (at the end of the epoch)
        if self.swa_handler is not None:
            self.swa_handler.update_average(self.model, epoch)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1} Training Loss: {avg_loss}")
        return avg_loss

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            tuple: (avg_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Containers for metric calculation
        all_targets = []
        all_preds = []
        all_identities = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                targets = batch["target"].to(self.device, non_blocking=True)
                aux_targets = batch["aux_targets"].to(self.device, non_blocking=True)

                # Validation might not have sample weights, default to 1.0 for loss calc
                if "weight" in batch:
                    weights = batch["weight"].to(self.device, non_blocking=True)
                else:
                    weights = torch.ones_like(targets).to(self.device)

                # Trim
                input_ids, attention_mask = self._trim_batch(input_ids, attention_mask)

                # Forward
                tox_logits, ident_logits = self.model(input_ids, attention_mask)

                # Loss
                loss = self.criterion(
                    tox_logits, ident_logits, targets, aux_targets, weights
                )
                total_loss += loss.item()
                num_batches += 1

                # Store predictions for metrics
                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(tox_logits).squeeze(-1)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_identities.append(aux_targets.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate all batches
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        identities = np.concatenate(all_identities)

        # Calculate Bias Metrics
        calculator = BiasMetricCalculator()
        metrics, _ = calculator.calculate_bias_metrics(y_true, y_pred, identities)

        print(f"Validation Loss: {avg_loss}")
        print(f"Validation Overall AUC: {metrics['overall_auc']}")
        print(f"Validation Final Score: {metrics['final_score']}")

        return avg_loss, metrics

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): The test data loader.

        Returns:
            dict: Dictionary mapping ID to predicted probability.
        """
        self.model.eval()
        predictions = {}

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                ids = batch["id"].cpu().numpy()

                # Trim
                input_ids, attention_mask = self._trim_batch(input_ids, attention_mask)

                # Forward (only need toxicity logits)
                tox_logits, _ = self.model(input_ids, attention_mask)

                # Probabilities
                preds = torch.sigmoid(tox_logits).squeeze(-1).cpu().numpy()

                # Store
                for id_val, pred_val in zip(ids, preds):
                    predictions[id_val] = float(pred_val)

        return predictions
