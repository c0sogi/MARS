import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import jaccard


class QATrainer:
    def __init__(self, model, tokenizer, device, optimizer=None, scheduler=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=Config.FP16)

    def train_epoch(self, data_loader, epoch_idx):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        losses = []

        # Progress bar
        pbar = tqdm(data_loader, desc=f"Train Epoch {epoch_idx}", leave=False)

        for batch in pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with Mixed Precision
            with autocast(enabled=Config.FP16):
                start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                # Calculate Loss
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            # Backward pass
            self.scaler.scale(total_loss).backward()

            # Clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            losses.append(total_loss.item())
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        return np.mean(losses)

    def eval_epoch(self, data_loader, raw_df):
        """
        Evaluates the model on the validation set.
        Computes Loss and Jaccard Score.
        """
        self.model.eval()
        losses = []
        jaccard_scores = []

        # Create a lookup for ground truth answers
        # raw_df should contain 'id' and 'answer_text'
        gt_lookup = raw_df.set_index("id")["answer_text"].to_dict()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)
                example_ids = batch["example_id"]

                # Forward pass
                with autocast(enabled=Config.FP16):
                    start_logits, end_logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                    loss_fct = nn.CrossEntropyLoss()
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    total_loss = (start_loss + end_loss) / 2

                losses.append(total_loss.item())

                # Decode predictions for Jaccard calculation
                pred_start_idxs = torch.argmax(start_logits, dim=1).cpu().numpy()
                pred_end_idxs = torch.argmax(end_logits, dim=1).cpu().numpy()
                input_ids_cpu = input_ids.cpu().numpy()

                for i, ex_id in enumerate(example_ids):
                    start_idx = pred_start_idxs[i]
                    end_idx = pred_end_idxs[i]

                    # Basic validity check
                    if start_idx > end_idx:
                        pred_text = ""
                    else:
                        # Decode token IDs to string
                        pred_text = self.tokenizer.decode(
                            input_ids_cpu[i][start_idx : end_idx + 1],
                            skip_special_tokens=True,
                        )

                    gt_text = gt_lookup.get(ex_id, "")
                    score = jaccard(gt_text, pred_text)
                    jaccard_scores.append(score)

        avg_loss = np.mean(losses)
        avg_jaccard = np.mean(jaccard_scores)

        print(f"Validation Loss: {avg_loss:.8f}")
        print(f"Validation Jaccard: {avg_jaccard:.8f}")

        return avg_jaccard, avg_loss

    def predict_and_submit(
        self, data_loader, raw_test_df, output_path=Config.SUBMISSION_PATH
    ):
        """
        Generates predictions for the test set and saves to submission.csv.
        Handles sliding window aggregation.
        """
        self.model.eval()

        # Store features per example_id
        # Key: example_id, Value: list of (start_logits, end_logits, offset_mapping, token_type_ids)
        all_results = collections.defaultdict(list)

        print("Running inference on test set...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Inference"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)

                # offset_mapping is a tensor of shape (batch, seq_len, 2)
                offset_mapping = batch["offset_mapping"].cpu().numpy()
                example_ids = batch["example_id"]

                with autocast(enabled=Config.FP16):
                    start_logits, end_logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                start_logits = start_logits.cpu().numpy()
                end_logits = end_logits.cpu().numpy()
                token_type_ids_cpu = token_type_ids.cpu().numpy()

                for i, ex_id in enumerate(example_ids):
                    all_results[ex_id].append(
                        {
                            "start_logits": start_logits[i],
                            "end_logits": end_logits[i],
                            "offsets": offset_mapping[i],
                            "token_type_ids": token_type_ids_cpu[i],
                        }
                    )

        # Process results to find best answers
        final_predictions = []

        print("Post-processing predictions...")
        for _, row in tqdm(raw_test_df.iterrows(), total=len(raw_test_df)):
            ex_id = row["id"]
            context_text = row["context"]

            if ex_id not in all_results:
                # Fallback
                final_predictions.append({"id": ex_id, "PredictionString": ""})
                continue

            features = all_results[ex_id]
            best_score = -float("inf")
            best_answer = ""

            for feature in features:
                start_logits = feature["start_logits"]
                end_logits = feature["end_logits"]
                offsets = feature["offsets"]
                token_type_ids = feature["token_type_ids"]

                # Get top-k start and end indices
                # We only consider tokens that are part of the context (token_type_ids == 1 for MuRIL context usually)
                # Note: MuRIL/BERT usually 0=Query, 1=Context.
                # We mask out non-context tokens by setting logits to -inf

                # Create a mask for context tokens (assuming 1 is context)
                # Also ensure we don't pick padding (offsets=(0,0) usually for special tokens)
                context_mask = token_type_ids == 1

                # Apply mask
                min_score = -1e9
                s_logits = np.where(context_mask, start_logits, min_score)
                e_logits = np.where(context_mask, end_logits, min_score)

                # Get top N indices
                start_indexes = np.argsort(s_logits)[-Config.N_BEST_SIZE :][::-1]
                end_indexes = np.argsort(e_logits)[-Config.N_BEST_SIZE :][::-1]

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Basic constraints
                        if start_index > end_index:
                            continue
                        if end_index - start_index + 1 > Config.MAX_ANSWER_LENGTH:
                            continue

                        score = start_logits[start_index] + end_logits[end_index]

                        if score > best_score:
                            best_score = score

                            # Map token indices to character indices
                            start_char = offsets[start_index][0]
                            end_char = offsets[end_index][1]

                            # Extract answer from original context
                            best_answer = context_text[start_char:end_char]

            # Quote the answer as per submission format requirement
            # "Note that the selected text needs to be quoted" -> The example shows "answer"
            # However, pandas to_csv with quoting=csv.QUOTE_NONNUMERIC or similar handles this.
            # The example shows: id,PredictionString \n 8c...,"1"
            # We will store the raw string, and ensure pandas quotes it.
            final_predictions.append({"id": ex_id, "PredictionString": best_answer})

        # Save submission
        sub_df = pd.DataFrame(final_predictions)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def train_model(model, tokenizer, train_loader, val_loader, raw_val_df, device):
    """
    Helper function to run the full training loop for a fold.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = QATrainer(model, tokenizer, device, optimizer, scheduler)

    best_jaccard = 0.0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        print(f"Train Loss: {train_loss:.8f}")

        val_jaccard, val_loss = trainer.eval_epoch(val_loader, raw_val_df)

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            # Save best model state
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
            print("New best model saved!")

    return best_jaccard
