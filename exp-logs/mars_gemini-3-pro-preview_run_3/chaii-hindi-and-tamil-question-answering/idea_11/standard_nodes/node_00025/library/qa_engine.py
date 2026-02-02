import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict

from library.config import Config
from library.utils import set_seed, compute_jaccard_score, ensure_dir
from library.data import prepare_qa_features, QADataset


class QAEngine:
    def __init__(self):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TAPT_MODEL_DIR)

    def get_optimizer(self, model):
        """
        Sets up the optimizer with weight decay correction.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)
        return optimizer

    def get_best_span(
        self, start_logits, end_logits, input_ids, offset_mapping, context_text
    ):
        """
        Extracts the best span from a single window based on logits.
        Returns (score, prediction_string).
        """
        # Identify context bounds based on sep_token_id (2 for XLM-R)
        # Format: <s> Q </s> </s> C </s>
        sep_indices = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[
            0
        ]

        if len(sep_indices) < 2:
            return -float("inf"), ""

        # Context starts after the second </s> (index 1 in 0-based list of seps)
        # And ends before the last </s>
        # If there are 3 seps: [sep_q, sep_sep, sep_end] -> context between sep_sep and sep_end
        # Usually XLM-R tokenizer(q, c) gives: <s> Q </s> </s> C </s>
        # Indices of </s>: [len_q, len_q+1, len_total-1]

        context_start_idx = sep_indices[1] + 1
        context_end_idx = sep_indices[-1] - 1

        if context_start_idx > context_end_idx:
            return -float("inf"), ""

        # Restrict logits to context
        valid_start_logits = start_logits[context_start_idx : context_end_idx + 1]
        valid_end_logits = end_logits[context_start_idx : context_end_idx + 1]

        if len(valid_start_logits) == 0:
            return -float("inf"), ""

        # Find best start/end pair
        # We limit the max answer length to something reasonable (e.g., 30 tokens) to speed up
        max_ans_len = 30

        best_score = -float("inf")
        best_start = -1
        best_end = -1

        # Vectorized approach for small window
        # Create a matrix of sums
        # We can just loop, it's small (max 384)

        # Get top K start and end indices to reduce complexity
        k = min(20, len(valid_start_logits))
        top_start_indices = torch.topk(valid_start_logits, k).indices
        top_end_indices = torch.topk(valid_end_logits, k).indices

        for s_idx in top_start_indices:
            for e_idx in top_end_indices:
                if e_idx >= s_idx and (e_idx - s_idx) < max_ans_len:
                    score = valid_start_logits[s_idx] + valid_end_logits[e_idx]
                    if score > best_score:
                        best_score = score
                        best_start = s_idx + context_start_idx
                        best_end = e_idx + context_start_idx

        if best_start == -1:
            return -float("inf"), ""

        # Extract text
        try:
            start_char = offset_mapping[best_start][0]
            end_char = offset_mapping[best_end][1]
            pred_text = context_text[start_char:end_char]
            return best_score.item(), pred_text
        except Exception:
            return -float("inf"), ""

    def train_one_epoch(self, model, dataloader, optimizer, epoch):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss}")
        return avg_loss

    def validate(self, model, dataloader, features_df, gt_map):
        """
        Validates the model using Jaccard score.
        Aggregates predictions across windows for each document.
        """
        model.eval()

        # Store best prediction per example_id
        # format: example_id -> (best_score, best_text)
        doc_preds = defaultdict(lambda: (-float("inf"), ""))

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                indices = batch["index"].cpu().numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Logits: [Batch, Seq, Num_Labels]
                # 0: O, 1: B-ANS, 2: I-ANS
                # We use B-ANS (1) as start score and B-ANS (1) or I-ANS (2) logic?
                # Standard approach: Start Logits = Label 1, End Logits = Label 1 or 2?
                # Actually, for simple span extraction, often people train Start/End heads.
                # But here we used TokenClassification with B/I/O.
                # Strategy:
                # Start Score = Logit(B-ANS)
                # End Score = Logit(I-ANS) (heuristic) or we just look for B followed by Is.
                # Let's use a robust heuristic for B/I/O:
                # Start score = Logit[B]
                # End score = Logit[I] (at end position)
                # This is an approximation. A B-token starts the span, an I-token (or B) ends it?
                # Let's stick to the prompt's implied simple span extraction or standard B/I/O decoding.
                # Given "Global Confidence Aggregation" usually implies Start/End logits.
                # With B/I/O, we can treat Logit(B) as start score and Logit(I) as continuation.
                # Let's define: Score(span) = Logit(B at start) + Sum(Logit(I) inside) ? Too complex.
                # Simplified: Start Logit = Logit(B), End Logit = Logit(I) (or B if len=1).

                logits = outputs.logits  # [B, L, 3]
                start_logits = logits[:, :, 1]  # B-ANS
                # For end logits, we can use a mix, but let's use I-ANS (2) as the signal for being part of answer
                # If single token answer, it's B. If multi, B I I.
                # Let's use: Start = B, End = I (or B).
                # To make it compatible with standard span logic:
                # We will just use the B-ANS logit for start.
                # We will use the I-ANS logit for end? No, I-ANS is high in the middle.
                # Let's strictly decode B-(I)* sequences.

                preds = torch.argmax(logits, dim=2).cpu().numpy()  # [B, L]

                # Iterate batch
                for i, idx in enumerate(indices):
                    row_idx = int(idx)
                    example_id = features_df.iloc[row_idx]["example_id"]
                    context = features_df.iloc[row_idx]["context"]
                    offset_mapping = features_df.iloc[row_idx]["offset_mapping"]

                    # Decode B-I-O
                    pred_labels = preds[i]
                    # Find spans
                    spans = []
                    current_start = -1
                    current_score = 0

                    for t_i, label in enumerate(pred_labels):
                        if label == 1:  # B-ANS
                            if current_start != -1:
                                # Close previous span
                                spans.append((current_start, t_i - 1, current_score))
                            current_start = t_i
                            current_score = logits[i, t_i, 1].item()
                        elif label == 2:  # I-ANS
                            if current_start != -1:
                                current_score += logits[i, t_i, 2].item()
                            else:
                                # I without B, ignore or treat as start? Strict: ignore.
                                pass
                        else:  # O
                            if current_start != -1:
                                spans.append((current_start, t_i - 1, current_score))
                                current_start = -1
                                current_score = 0

                    if current_start != -1:
                        spans.append(
                            (current_start, len(pred_labels) - 1, current_score)
                        )

                    # Process spans to find best for this window
                    for start, end, score in spans:
                        # Map to char offsets
                        try:
                            # Check if valid context offsets (not (0,0) unless it's the very start)
                            # XLM-R offsets are usually good.
                            s_char = offset_mapping[start][0]
                            e_char = offset_mapping[end][1]
                            if s_char == 0 and e_char == 0 and start != 0:
                                continue  # Special token

                            text = context[s_char:e_char]

                            # Normalize score by length to avoid bias towards long answers?
                            # Or just raw sum. Prompt says "highest confidence score".
                            # Let's use average logit to be length invariant.
                            avg_score = score / (end - start + 1)

                            if avg_score > doc_preds[example_id][0]:
                                doc_preds[example_id] = (avg_score, text)
                        except:
                            continue

        # Compute Jaccard
        ground_truths = []
        predictions = []

        for eid, (score, text) in doc_preds.items():
            if eid in gt_map:
                ground_truths.append(gt_map[eid])
                predictions.append(text)

        score = compute_jaccard_score(ground_truths, predictions)
        print(f"Validation Jaccard: {score}")
        return score

    def inference(self, model, dataloader, features_df):
        """
        Runs inference on the test set.
        Returns a dict: example_id -> prediction_string
        """
        model.eval()
        doc_preds = defaultdict(lambda: (-float("inf"), ""))

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                indices = batch["index"].cpu().numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2).cpu().numpy()

                for i, idx in enumerate(indices):
                    row_idx = int(idx)
                    example_id = features_df.iloc[row_idx]["example_id"]
                    context = features_df.iloc[row_idx]["context"]
                    offset_mapping = features_df.iloc[row_idx]["offset_mapping"]

                    pred_labels = preds[i]
                    spans = []
                    current_start = -1
                    current_score = 0

                    for t_i, label in enumerate(pred_labels):
                        if label == 1:  # B-ANS
                            if current_start != -1:
                                spans.append((current_start, t_i - 1, current_score))
                            current_start = t_i
                            current_score = logits[i, t_i, 1].item()
                        elif label == 2:  # I-ANS
                            if current_start != -1:
                                current_score += logits[i, t_i, 2].item()
                        else:  # O
                            if current_start != -1:
                                spans.append((current_start, t_i - 1, current_score))
                                current_start = -1
                                current_score = 0
                    if current_start != -1:
                        spans.append(
                            (current_start, len(pred_labels) - 1, current_score)
                        )

                    for start, end, score in spans:
                        try:
                            s_char = offset_mapping[start][0]
                            e_char = offset_mapping[end][1]
                            if s_char == 0 and e_char == 0 and start != 0:
                                continue
                            text = context[s_char:e_char]
                            avg_score = score / (end - start + 1)

                            if avg_score > doc_preds[example_id][0]:
                                doc_preds[example_id] = (avg_score, text)
                        except:
                            continue

        # Convert to dict
        final_preds = {}
        # Ensure we have an entry for every ID in features
        all_ids = features_df["example_id"].unique()
        for eid in all_ids:
            if eid in doc_preds:
                final_preds[eid] = doc_preds[eid][1]
            else:
                final_preds[eid] = ""  # Default to empty

        return final_preds

    def run_k_fold_training(self):
        set_seed(Config.SEED)

        # 1. Prepare Combined Data
        print("Preparing combined training data...")
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_combined = pd.concat([df_train, df_val], ignore_index=True)

        combined_path = os.path.join(Config.WORKING_DIR, "combined_train.csv")
        df_combined.to_csv(combined_path, index=False)

        # 2. Generate Features
        train_features = prepare_qa_features(
            self.tokenizer,
            combined_path,
            "train_full",
            load_cached_data=True,
            is_test=False,
        )

        # 3. Stratified K-Fold Split
        # We split based on unique example_ids to prevent leakage
        unique_examples = df_combined[["id", "language"]].drop_duplicates()
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_splits = []
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(unique_examples, unique_examples["language"])
        ):
            train_ids = set(unique_examples.iloc[train_idx]["id"])
            val_ids = set(unique_examples.iloc[val_idx]["id"])
            fold_splits.append((train_ids, val_ids))

        # 4. Training Loop
        for fold, (train_ids, val_ids) in enumerate(fold_splits):
            print(f"\n=== Training Fold {fold + 1}/{Config.N_FOLDS} ===")

            # Filter features
            train_fold_df = train_features[
                train_features["example_id"].isin(train_ids)
            ].reset_index(drop=True)
            val_fold_df = train_features[
                train_features["example_id"].isin(val_ids)
            ].reset_index(drop=True)

            # Create Datasets
            train_dataset = QADataset(train_fold_df, is_test=False)
            val_dataset = QADataset(val_fold_df, is_test=False)

            train_loader = DataLoader(
                train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
            )

            # Init Model
            model = AutoModelForTokenClassification.from_pretrained(
                Config.TAPT_MODEL_DIR, num_labels=3
            )
            model.to(self.device)
            optimizer = self.get_optimizer(model)

            # Ground Truth Map for Validation
            gt_map = (
                df_combined[df_combined["id"].isin(val_ids)]
                .set_index("id")["answer_text"]
                .to_dict()
            )

            best_jaccard = -1.0
            model_save_path = os.path.join(Config.QA_MODEL_DIR, f"model_fold_{fold}.pt")

            for epoch in range(Config.EPOCHS):
                self.train_one_epoch(model, train_loader, optimizer, epoch)
                val_jaccard = self.validate(model, val_loader, val_fold_df, gt_map)

                if val_jaccard > best_jaccard:
                    best_jaccard = val_jaccard
                    print(f"New best Jaccard: {best_jaccard}. Saving model...")
                    torch.save(model.state_dict(), model_save_path)

            # Cleanup
            del model, optimizer, train_loader, val_loader
            torch.cuda.empty_cache()

        print("K-Fold Training Complete.")

    def predict_and_submit(self):
        print("\nStarting Inference and Submission...")

        # 1. Load Test Features
        test_features = prepare_qa_features(
            self.tokenizer,
            Config.TEST_META_PATH,
            "test",
            load_cached_data=True,
            is_test=True,
        )
        test_dataset = QADataset(test_features, is_test=True)
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # 2. Collect Predictions from all folds
        all_fold_preds = []  # List of dicts

        for fold in range(Config.N_FOLDS):
            print(f"Inference Fold {fold + 1}...")
            model_path = os.path.join(Config.QA_MODEL_DIR, f"model_fold_{fold}.pt")

            model = AutoModelForTokenClassification.from_pretrained(
                Config.TAPT_MODEL_DIR, num_labels=3
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)

            fold_preds = self.inference(model, test_loader, test_features)
            all_fold_preds.append(fold_preds)

            del model
            torch.cuda.empty_cache()

        # 3. Majority Voting
        print("Performing Majority Voting...")
        final_submission = []
        test_ids = test_features["example_id"].unique()

        for eid in test_ids:
            candidates = [fold_preds.get(eid, "") for fold_preds in all_fold_preds]
            # Count votes
            vote_counts = Counter(candidates)
            # Get most common
            best_pred, _ = vote_counts.most_common(1)[0]

            # Format strictly as quoted string
            final_submission.append({"id": eid, "PredictionString": f'"{best_pred}"'})

        # 4. Save Submission
        df_sub = pd.DataFrame(final_submission)
        # Ensure columns match sample submission
        df_sub = df_sub[["id", "PredictionString"]]

        ensure_dir(Config.SUBMISSION_FILE)
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(df_sub.head())


def run_k_fold_training():
    engine = QAEngine()
    engine.run_k_fold_training()


def predict_and_submit():
    engine = QAEngine()
    engine.predict_and_submit()
