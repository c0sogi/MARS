import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from library.config import Config
from library.utils import set_seed, ensure_dir
from library.data import prepare_qa_features, QADataset


class InferenceEngine:
    """
    Engine for running inference using the trained QA models.
    Implements sliding window aggregation and ensemble voting.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TAPT_MODEL_DIR)

    def predict_document(self, model, dataloader, features_df):
        """
        Aggregates predictions from sliding windows of a document.
        Selects the span with the highest global confidence score.

        Args:
            model: The trained TokenClassification model.
            dataloader: DataLoader for the test set features.
            features_df: DataFrame containing feature metadata (context, offsets).

        Returns:
            Dict[str, str]: Mapping from example_id to predicted answer string.
        """
        model.eval()

        # Dictionary to store the best prediction for each document
        # Key: example_id, Value: (score, text)
        doc_preds = defaultdict(lambda: (-float("inf"), ""))

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                indices = batch["index"].cpu().numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Shape: [Batch, SeqLen, 3]

                # Get predicted class indices (0=O, 1=B-ANS, 2=I-ANS)
                preds = torch.argmax(logits, dim=2).cpu().numpy()

                for i, idx in enumerate(indices):
                    row_idx = int(idx)
                    example_id = features_df.iloc[row_idx]["example_id"]
                    context = features_df.iloc[row_idx]["context"]
                    offset_mapping = features_df.iloc[row_idx]["offset_mapping"]

                    pred_labels = preds[i]

                    # Extract spans from the sequence (B-ANS followed by optional I-ANS)
                    spans = []
                    current_start = -1
                    current_score = 0.0

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
                                # Ignore I-ANS without preceding B-ANS (Strict Containment logic)
                                pass
                        else:  # O
                            if current_start != -1:
                                spans.append((current_start, t_i - 1, current_score))
                                current_start = -1
                                current_score = 0.0

                    # Handle span at the end of sequence
                    if current_start != -1:
                        spans.append(
                            (current_start, len(pred_labels) - 1, current_score)
                        )

                    # Process extracted spans
                    for start, end, score in spans:
                        try:
                            # Retrieve character offsets
                            s_char = offset_mapping[start][0]
                            e_char = offset_mapping[end][1]

                            # Skip special tokens (often mapped to 0,0)
                            if s_char == 0 and e_char == 0 and start != 0:
                                continue

                            pred_text = context[s_char:e_char]

                            # Calculate confidence (Average Logit)
                            # Normalizing by length prevents bias towards very long answers
                            span_len = end - start + 1
                            if span_len > 0:
                                avg_score = score / span_len

                                # Update global best for this document
                                if avg_score > doc_preds[example_id][0]:
                                    doc_preds[example_id] = (avg_score, pred_text)
                        except (IndexError, TypeError):
                            continue

        # Extract just the text for the return value
        final_preds = {}
        all_ids = features_df["example_id"].unique()
        for eid in all_ids:
            if eid in doc_preds:
                final_preds[eid] = doc_preds[eid][1]
            else:
                final_preds[eid] = ""

        return final_preds

    def majority_vote(self, all_fold_preds):
        """
        Applies majority voting to predictions from multiple models.

        Args:
            all_fold_preds: List of dicts, where each dict is {example_id: prediction_string}

        Returns:
            Dict[str, str]: Consolidated predictions {example_id: prediction_string}
        """
        final_preds = {}

        # Collect all unique example IDs
        all_ids = set()
        for preds in all_fold_preds:
            all_ids.update(preds.keys())

        for eid in all_ids:
            # Gather predictions for this ID from all folds
            candidates = [fold_preds.get(eid, "") for fold_preds in all_fold_preds]

            # Count occurrences
            vote_counts = Counter(candidates)

            # Select the most common prediction
            # most_common(1) returns [(value, count)]
            best_pred = vote_counts.most_common(1)[0][0]
            final_preds[eid] = best_pred

        return final_preds

    def ensemble_predict(self):
        """
        Main inference pipeline:
        1. Prepares test features.
        2. Runs inference for each K-fold model.
        3. Aggregates results via majority voting.
        4. Saves submission file.
        """
        set_seed(Config.SEED)
        print("Starting Ensemble Inference...")

        # 1. Prepare Test Data
        test_features = prepare_qa_features(
            self.tokenizer,
            Config.TEST_META_PATH,
            "test",
            load_cached_data=True,
            is_test=True,
        )

        test_dataset = QADataset(test_features, is_test=True)
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )

        all_fold_preds = []

        # 2. Iterate through Folds
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.QA_MODEL_DIR, f"model_fold_{fold}.pt")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
                )
                continue

            print(f"Loading model for Fold {fold}...")
            # Initialize model architecture
            model = AutoModelForTokenClassification.from_pretrained(
                Config.TAPT_MODEL_DIR, num_labels=3
            )
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)

            print(f"Running inference for Fold {fold}...")
            fold_preds = self.predict_document(model, test_loader, test_features)
            all_fold_preds.append(fold_preds)

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()

        if not all_fold_preds:
            print(
                "Error: No models were successfully loaded. Generating empty submission."
            )
            final_preds = {eid: "" for eid in test_features["example_id"].unique()}
        else:
            # 3. Apply Majority Voting
            print("Applying Majority Voting...")
            final_preds = self.majority_vote(all_fold_preds)

        # 4. Generate Submission
        print("Generating submission file...")
        submission_rows = []
        for eid, pred_str in final_preds.items():
            submission_rows.append({"id": eid, "PredictionString": pred_str})

        df_sub = pd.DataFrame(submission_rows)

        # Ensure correct column order
        df_sub = df_sub[["id", "PredictionString"]]

        ensure_dir(Config.SUBMISSION_FILE)
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print("First 5 rows:")
        print(df_sub.head())
