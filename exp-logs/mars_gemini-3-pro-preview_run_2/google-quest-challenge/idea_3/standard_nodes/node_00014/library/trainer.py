import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config
from library.model import CausalAwareSiameseDeberta
from library.dataset import get_dataloaders
from library.metrics import compute_spearmanr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Trainer:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Model
        self.model = CausalAwareSiameseDeberta()
        self.model.to(self.device)

        # Loss Function
        # We use BCEWithLogitsLoss because the model outputs logits and targets are [0,1] probabilities
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer with Differential Learning Rates
        # Backbone gets lower LR, Heads get higher LR
        optimizer_grouped_parameters = [
            {"params": self.model.backbone.parameters(), "lr": self.cfg.LR_BACKBONE},
            {"params": self.model.question_head.parameters(), "lr": self.cfg.LR_HEAD},
            {"params": self.model.answer_head.parameters(), "lr": self.cfg.LR_HEAD},
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.LR_HEAD,  # Base LR (overridden by groups)
            weight_decay=self.cfg.WEIGHT_DECAY,
        )

    def train_one_epoch(self, dataloader, scheduler, epoch_idx):
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Move batch to device
            inputs = {
                "input_ids_q": batch["input_ids_q"].to(self.device),
                "attention_mask_q": batch["attention_mask_q"].to(self.device),
                "input_ids_a": batch["input_ids_a"].to(self.device),
                "attention_mask_a": batch["attention_mask_a"].to(self.device),
            }

            if "token_type_ids_q" in batch:
                inputs["token_type_ids_q"] = batch["token_type_ids_q"].to(self.device)
            if "token_type_ids_a" in batch:
                inputs["token_type_ids_a"] = batch["token_type_ids_a"].to(self.device)

            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(**inputs)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids_q": batch["input_ids_q"].to(self.device),
                    "attention_mask_q": batch["attention_mask_q"].to(self.device),
                    "input_ids_a": batch["input_ids_a"].to(self.device),
                    "attention_mask_a": batch["attention_mask_a"].to(self.device),
                }

                if "token_type_ids_q" in batch:
                    inputs["token_type_ids_q"] = batch["token_type_ids_q"].to(
                        self.device
                    )
                if "token_type_ids_a" in batch:
                    inputs["token_type_ids_a"] = batch["token_type_ids_a"].to(
                        self.device
                    )

                labels = batch["labels"].to(self.device)

                logits = self.model(**inputs)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(labels_list, axis=0)

        score = compute_spearmanr(targets, preds)
        return score

    def fit(self):
        print("Initializing Tokenizer and Dataloaders...")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_NAME)
        train_loader, val_loader, test_loader = get_dataloaders(
            tokenizer, load_cached_data=True
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.cfg.EPOCHS
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        best_score = -1.0
        patience_counter = 0
        best_model_path = os.path.join(self.cfg.WORKING_DIR, "best_model.pth")

        print(f"Starting training for {self.cfg.EPOCHS} epochs...")

        for epoch in range(self.cfg.EPOCHS):
            train_loss = self.train_one_epoch(train_loader, scheduler, epoch)
            val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{self.cfg.EPOCHS} | Train Loss: {train_loss:.6f} | Val Spearman: {val_score}"
            )

            if val_score > best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with score: {best_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load best model for inference
        print(f"Loading best model from {best_model_path}...")
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

        return test_loader

    def predict(self, test_loader):
        self.model.eval()
        preds_list = []
        qa_ids_list = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                    "input_ids_q": batch["input_ids_q"].to(self.device),
                    "attention_mask_q": batch["attention_mask_q"].to(self.device),
                    "input_ids_a": batch["input_ids_a"].to(self.device),
                    "attention_mask_a": batch["attention_mask_a"].to(self.device),
                }

                if "token_type_ids_q" in batch:
                    inputs["token_type_ids_q"] = batch["token_type_ids_q"].to(
                        self.device
                    )
                if "token_type_ids_a" in batch:
                    inputs["token_type_ids_a"] = batch["token_type_ids_a"].to(
                        self.device
                    )

                qa_ids = batch["qa_id"]

                logits = self.model(**inputs)
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                qa_ids_list.extend(qa_ids.numpy())

        all_preds = np.concatenate(preds_list, axis=0)
        all_qa_ids = np.array(qa_ids_list)

        return all_qa_ids, all_preds

    def generate_submission(self, qa_ids, preds):
        # Create submission directory
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        # Prepare DataFrame
        # Columns must match sample_submission.csv
        # The model outputs are concatenated [Question Targets, Answer Targets]
        # which matches the order in Config.TARGET_COLS

        df_sub = pd.DataFrame(preds, columns=self.cfg.TARGET_COLS)
        df_sub.insert(0, "qa_id", qa_ids)

        # Ensure qa_id is integer
        df_sub["qa_id"] = df_sub["qa_id"].astype(int)

        # Save
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")


def main():
    # Set seed for reproducibility
    cfg = Config()
    set_seed(cfg.SEED)

    # Initialize Trainer
    trainer = Trainer()

    # Train and Validate
    test_loader = trainer.fit()

    # Predict
    qa_ids, preds = trainer.predict(test_loader)

    # Save Submission
    trainer.generate_submission(qa_ids, preds)


if __name__ == "__main__":
    main()
