import os
import torch
import pandas as pd
from torch.optim import AdamW
from library.config import Config
from library.utils import jaccard, find_best_substring
from library.model import load_model


class Trainer:
    """
    Trainer class for the Hindi/Tamil Question Answering task.
    Handles training, validation, and inference.
    """

    def __init__(self, model, tokenizer, device=None):
        """
        Args:
            model: The PyTorch model (MT5ForConditionalGeneration).
            tokenizer: The tokenizer (AutoTokenizer).
            device: torch.device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else Config.DEVICE

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Optimization step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Accumulate loss
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        return total_loss / count if count > 0 else 0.0

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set using Jaccard score.
        Aggregates sliding window predictions by ID before scoring.
        """
        self.model.eval()

        pred_dict = {}  # id -> list of predictions
        gt_dict = {}  # id -> ground truth answer

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                contexts = batch["context"]
                ids = batch["ids"]

                # Check if ground truth is available in the batch
                has_gt = "answer_text" in batch
                if has_gt:
                    gts = batch["answer_text"]

                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=Config.MAX_TARGET_LENGTH,
                )

                # Decode predictions
                decoded_preds = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Post-processing and collection
                for i, pred_text in enumerate(decoded_preds):
                    sample_id = ids[i]
                    context = contexts[i]

                    # Post-processing to find best substring in context
                    final_pred = find_best_substring(context, pred_text)

                    if sample_id not in pred_dict:
                        pred_dict[sample_id] = []
                    pred_dict[sample_id].append(final_pred)

                    if has_gt:
                        # Ground truth is same for all windows of the same ID
                        gt_dict[sample_id] = gts[i]

        # Aggregate and Calculate Score
        total_jaccard = 0.0
        count = 0

        for sample_id, preds in pred_dict.items():
            # Aggregation Strategy: Longest string (matches inference logic)
            best_pred = max(preds, key=len) if preds else ""

            # Get Ground Truth
            gt_text = gt_dict.get(sample_id, "")

            score = jaccard(gt_text, best_pred)
            total_jaccard += score
            count += 1

        return total_jaccard / count if count > 0 else 0.0

    def fit(self, train_loader, val_loader):
        """
        Runs the full training loop with early stopping.
        """
        best_score = -1.0
        patience_counter = 0

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_score = self.evaluate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Jaccard: {val_score}"
            )

            # Check for improvement
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0

                # Save the best model
                # Ensure directory exists
                os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
                self.model.save_pretrained(Config.MODEL_SAVE_PATH)
                self.tokenizer.save_pretrained(Config.TOKENIZER_SAVE_PATH)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves to submission file.
        Reloads the best model found during training.
        """
        # Reload the best model if it exists
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print("Loading best model for prediction...")
            self.model = load_model(Config.MODEL_SAVE_PATH, self.device)
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()
        results = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                ids = batch["ids"]
                contexts = batch["context"]

                # Generate
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=Config.MAX_TARGET_LENGTH,
                )

                # Decode
                decoded_preds = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Post-process and collect results
                for i, pred_text in enumerate(decoded_preds):
                    context = contexts[i]
                    sample_id = ids[i]

                    final_pred = find_best_substring(context, pred_text)

                    results.append({"id": sample_id, "PredictionString": final_pred})

        # Create DataFrame and save
        df = pd.DataFrame(results)

        # Aggregate sliding window predictions
        df["pred_len"] = df["PredictionString"].astype(str).str.len()
        df_sorted = df.sort_values(by="pred_len", ascending=False)
        submission_df = df_sorted.groupby("id").first().reset_index()
        submission_df = submission_df[["id", "PredictionString"]]

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission file saved to {Config.SUBMISSION_FILE}")
