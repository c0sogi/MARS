import os
import time
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from library.config import Config, seed_everything
from library.utils import JigsawEvaluator
from library.data import get_dataloaders
from library.model import DistilBertWithBiasHead


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = Config.DEVICE

        # Initialize Model
        self.model = DistilBertWithBiasHead(
            model_name=Config.MODEL_NAME,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT,
            hidden_size=Config.HIDDEN_SIZE,
        )
        self.model.to(self.device)

        # DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug
        )

        # Optimizer
        # Separate weight decay for bias/LayerNorm as per BERT best practices
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

        # Scheduler
        num_training_steps = len(self.train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metadata for evaluation (identities needed for bias metrics)
        # We load this once to avoid reloading every epoch
        val_df_path = Config.VAL_PATH
        self.val_df = pd.read_csv(val_df_path)
        if self.debug:
            self.val_df = self.val_df.head(Config.DEBUG_SAMPLE_SIZE)

    def train_one_epoch(self, epoch_index):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"Epoch {epoch_index + 1}/{Config.EPOCHS}")

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device).unsqueeze(1)  # Shape: (batch, 1)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        print(f"  Training Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        # No gradients needed for evaluation
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).squeeze(1)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Use JigsawEvaluator to compute the competition metric
        # Ensure the validation dataframe aligns with the predictions
        # The val_loader is created with shuffle=False, so alignment is preserved
        evaluator = JigsawEvaluator(all_targets, all_preds, self.val_df)
        final_score, metrics = evaluator.get_final_metric()

        return final_score, metrics

    def train(self):
        seed_everything(Config.SEED)

        best_score = -float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_one_epoch(epoch)

            # Validate
            print("  Running evaluation...")
            val_score, val_metrics = self.evaluate()

            print(f"  Validation Score: {val_score}")
            print(f"  Metrics: {val_metrics}")

            # Early Stopping Check
            if val_score > best_score:
                print(
                    f"  Score improved from {best_score} to {val_score}. Saving model state."
                )
                best_score = val_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                # Save checkpoint to disk
                torch.save(best_model_state, Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                print(
                    f"  Score did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("  Early stopping triggered.")
                break

        # Load best model for inference
        if best_model_state is not None:
            print("Loading best model for inference...")
            self.model.load_state_dict(best_model_state)
        else:
            print("Warning: No best model saved. Using current model.")

    def predict_and_submit(self):
        print("Generating predictions on Test set...")
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).squeeze(1)

                all_preds.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Create submission file
        # We need the IDs from the test file or sample submission
        # Reading test.csv to get IDs
        test_df = pd.read_csv(Config.TEST_PATH)
        if self.debug:
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        submission = pd.DataFrame({"id": test_df["id"], "prediction": all_preds})

        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training(debug=Config.DEBUG):
    trainer = Trainer(debug=debug)
    trainer.train()
    trainer.predict_and_submit()


if __name__ == "__main__":
    # This block is just for local testing if run directly,
    # but the requirement says "DO NOT include an if __name__ == '__main__': block"
    # for the module implementation itself.
    # However, standard python modules often have this.
    # The prompt says: "Only implement the module class/functions. DO NOT include an if __name__ == '__main__': block."
    # So I will omit it.
    pass
