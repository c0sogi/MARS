import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.utils import JigsawEvaluator, seed_everything


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device=None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Loss function: Reduction='none' to apply sample weights manually
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Evaluator
        self.evaluator = JigsawEvaluator(
            identity_columns=config.IDENTITY_COLUMNS,
            weight_overall=0.25,
            weight_subgroup=0.25,
            weight_bpsn=0.25,
            weight_bnsp=0.25,
            power=-5,
        )

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)
            weights = batch["weight"].to(self.device)
            aux_targets = batch["aux_target"].to(self.device)

            # Device-Side Trimming: Slice to max length in this batch
            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            self.optimizer.zero_grad()

            # Forward pass
            tox_logits, ident_logits = self.model(input_ids, attention_mask)

            # Calculate Losses
            # Primary Task: Weighted BCE
            loss_tox = (self.criterion(tox_logits.view(-1), targets) * weights).mean()

            # Auxiliary Task: Standard BCE (averaged)
            loss_aux = self.criterion(ident_logits, aux_targets).mean()

            # Composite Loss
            loss = loss_tox + self.config.AUX_LOSS_WEIGHT * loss_aux

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch_idx} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self, val_identities_arr):
        """
        Runs validation and calculates bias metrics.
        val_identities_arr: Numpy array containing [identities..., target]
        """
        self.model.eval()
        preds_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Device-Side Trimming
                max_len = attention_mask.sum(dim=1).max().item()
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]

                tox_logits, _ = self.model(input_ids, attention_mask)
                batch_preds = torch.sigmoid(tox_logits).view(-1).cpu().numpy()
                preds_list.extend(batch_preds)

        y_pred = np.array(preds_list)

        # Reconstruct DataFrame for Evaluator
        # Columns: [identities..., target]
        cols = self.config.IDENTITY_COLUMNS + [self.config.TARGET_COL]
        val_df = pd.DataFrame(val_identities_arr, columns=cols)

        y_true = val_df[self.config.TARGET_COL].values
        identities_df = val_df[self.config.IDENTITY_COLUMNS]

        score, metrics = self.evaluator.get_final_metric(y_true, y_pred, identities_df)

        print(f"Validation Score: {score:.6f}")
        print(f"Overall AUC:      {metrics['overall_auc']:.6f}")
        print(f"Subgroup AUC:     {metrics['subgroup_auc']:.6f}")
        print(f"BPSN AUC:         {metrics['bpsn_auc']:.6f}")
        print(f"BNSP AUC:         {metrics['bnsp_auc']:.6f}")

        return score

    def save_checkpoint(self, filename="best_model.pth"):
        path = os.path.join(self.config.WORKING_DIR, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, filename="best_model.pth"):
        path = os.path.join(self.config.WORKING_DIR, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

    def fit(self, val_identities_arr, patience=2):
        best_score = -float("inf")
        no_improve_epochs = 0

        print("Starting training...")
        for epoch in range(1, self.config.EPOCHS + 1):
            self.train_epoch(epoch)
            score = self.validate(val_identities_arr)

            if score > best_score:
                best_score = score
                self.save_checkpoint("best_model.pth")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Best Validation Score: {best_score:.6f}")

    def predict(self, test_loader, output_ids):
        """
        Generates predictions for the test set and saves submission.
        """
        self.load_checkpoint("best_model.pth")
        self.model.eval()
        preds_list = []

        print("Generating test predictions...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Device-Side Trimming
                max_len = attention_mask.sum(dim=1).max().item()
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]

                tox_logits, _ = self.model(input_ids, attention_mask)
                batch_preds = torch.sigmoid(tox_logits).view(-1).cpu().numpy()
                preds_list.extend(batch_preds)

        # Create submission file
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")

        submission = pd.DataFrame({"id": output_ids, "prediction": preds_list})
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        return submission
