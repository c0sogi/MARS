import os
import collections
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import compute_average_jaccard


class TrainRunner:
    """
    Manages the training, validation, and inference lifecycle for a single fold
    of the Question Answering model.
    """

    def __init__(self, model, tokenizer, optimizer, scheduler, device):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.amp.GradScaler("cuda")

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0
        count = 0

        for batch in train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                loss = outputs.loss

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.max_grad_norm
            )

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else 0.0
        return avg_loss

    def validate(self, val_loader, val_features_df, raw_val_df):
        """
        Evaluates the model on the validation set using Jaccard score.
        Handles the mapping from sliding window logits to original text spans.
        """
        self.model.eval()
        all_start_logits = []
        all_end_logits = []

        # 1. Inference Loop
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )

                all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
                all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

        if len(all_start_logits) == 0:
            return 0.0

        all_start_logits = np.concatenate(all_start_logits, axis=0)
        all_end_logits = np.concatenate(all_end_logits, axis=0)

        # 2. Post-processing: Map logits to text
        # Group features by example_id
        example_to_features = collections.defaultdict(list)
        for idx, row in val_features_df.iterrows():
            example_to_features[row["example_id"]].append(idx)

        # Create a lookup for raw context
        raw_context_map = raw_val_df.set_index("id")["context"].to_dict()

        predictions = []
        ground_truths = []

        # Iterate in the order of raw_val_df to ensure alignment
        for _, row in raw_val_df.iterrows():
            ex_id = row["id"]
            ground_truths.append(row["answer_text"])

            if ex_id not in example_to_features:
                predictions.append("")
                continue

            feature_indices = example_to_features[ex_id]
            context_text = raw_context_map[ex_id]

            best_score = float("-inf")
            best_answer = ""

            # Search for best span across all chunks for this example
            for feat_idx in feature_indices:
                start_logits = all_start_logits[feat_idx]
                end_logits = all_end_logits[feat_idx]
                offsets = val_features_df.iloc[feat_idx]["offset_mapping"]

                # Get top-k start and end indices
                start_indexes = np.argsort(start_logits)[
                    -1 : -Config.n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -Config.n_best_size - 1 : -1
                ].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip invalid spans
                        if start_index >= len(offsets) or end_index >= len(offsets):
                            continue
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if end_index < start_index:
                            continue
                        if end_index - start_index + 1 > Config.max_answer_length:
                            continue

                        score = start_logits[start_index] + end_logits[end_index]

                        if score > best_score:
                            best_score = score
                            # Extract text using character offsets
                            try:
                                # offsets is list of [start, end]
                                start_char = offsets[start_index][0]
                                end_char = offsets[end_index][1]
                                best_answer = context_text[start_char:end_char]
                            except Exception:
                                pass

            predictions.append(best_answer)

        # 3. Compute Metric
        score = compute_average_jaccard(ground_truths, predictions)
        return score

    def run(self, train_loader, val_loader, val_features_df, raw_val_df, fold_idx):
        """
        Executes the full training loop with early stopping.
        """
        print(f"Starting training for fold {fold_idx}...")

        best_val_score = -1.0
        patience = 0
        patience_limit = 2  # Strict patience for small dataset/epochs

        best_model_path = os.path.join(
            Config.output_dir, f"best_model_fold_{fold_idx}.pth"
        )

        for epoch in range(Config.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_score = self.validate(val_loader, val_features_df, raw_val_df)

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss} | Val Jaccard: {val_score}"
            )

            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience += 1
                if patience >= patience_limit:
                    print("Early stopping triggered.")
                    break

        return best_val_score

    def predict(self, test_loader, test_features_df, raw_test_df):
        """
        Generates predictions for the test set.
        Returns a dictionary {id: prediction_string}.
        """
        self.model.eval()
        all_start_logits = []
        all_end_logits = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )

                all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
                all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

        if len(all_start_logits) > 0:
            all_start_logits = np.concatenate(all_start_logits, axis=0)
            all_end_logits = np.concatenate(all_end_logits, axis=0)

        # Map features to examples
        example_to_features = collections.defaultdict(list)
        for idx, row in test_features_df.iterrows():
            example_to_features[row["example_id"]].append(idx)

        raw_context_map = raw_test_df.set_index("id")["context"].to_dict()
        results = {}

        for ex_id, feature_indices in example_to_features.items():
            context_text = raw_context_map.get(ex_id, "")

            best_score = float("-inf")
            best_answer = ""

            for feat_idx in feature_indices:
                start_logits = all_start_logits[feat_idx]
                end_logits = all_end_logits[feat_idx]
                offsets = test_features_df.iloc[feat_idx]["offset_mapping"]

                start_indexes = np.argsort(start_logits)[
                    -1 : -Config.n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -Config.n_best_size - 1 : -1
                ].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(offsets) or end_index >= len(offsets):
                            continue
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if end_index < start_index:
                            continue
                        if end_index - start_index + 1 > Config.max_answer_length:
                            continue

                        score = start_logits[start_index] + end_logits[end_index]
                        if score > best_score:
                            best_score = score
                            try:
                                start_char = offsets[start_index][0]
                                end_char = offsets[end_index][1]
                                best_answer = context_text[start_char:end_char]
                            except Exception:
                                pass

            results[ex_id] = best_answer

        return results
