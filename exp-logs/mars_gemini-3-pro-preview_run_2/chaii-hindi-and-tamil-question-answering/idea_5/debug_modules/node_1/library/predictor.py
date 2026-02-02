import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

from library.config import Config
from library.model import MuRILForQA
from library.data_loader import get_processed_data, QADataset
from library.utils import seed_everything


class Predictor:
    def __init__(self):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
        seed_everything(Config.SEED)

    def predict_fn(self, model, data_loader):
        """
        Runs inference on the provided data_loader using the given model.
        Returns start and end logits as numpy arrays.
        """
        model.eval()
        model.to(self.device)

        all_start_logits = []
        all_end_logits = []

        # Disable gradients for inference
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)

                # Use mixed precision for inference
                with autocast(enabled=Config.FP16):
                    start_logits, end_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())

        # Concatenate all batches to form full dataset logits
        if len(all_start_logits) > 0:
            return np.concatenate(all_start_logits, axis=0), np.concatenate(
                all_end_logits, axis=0
            )
        else:
            return np.array([]), np.array([])

    def get_ensemble_predictions(self, folds=Config.NUM_FOLDS):
        """
        Loads models for each fold, aggregates predictions via averaging,
        post-processes spans, and saves the submission file.
        """
        print(f"Starting ensemble prediction with {folds} folds...")

        # 1. Load and Process Test Data
        # We use the original test path; metadata/test.csv is identical in content
        test_df = pd.read_csv(Config.TEST_PATH)

        # Process features (tokenization + sliding window)
        # load_cached_data=True ensures we don't recompute if already done
        features_df = get_processed_data(
            test_df, self.tokenizer, split="test", load_cached_data=True
        )

        # 2. Create DataLoader
        test_dataset = QADataset(features_df, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Initialize Aggregated Logits
        num_features = len(features_df)
        agg_start_logits = np.zeros((num_features, Config.MAX_LENGTH), dtype=np.float32)
        agg_end_logits = np.zeros((num_features, Config.MAX_LENGTH), dtype=np.float32)

        models_found = 0

        # 4. Ensemble Loop: Accumulate Logits
        for fold in range(folds):
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model checkpoint not found at {model_path}. Skipping fold {fold}."
                )
                continue

            print(f"Loading model for fold {fold} from {model_path}...")
            model = MuRILForQA()
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)

            start_logits, end_logits = self.predict_fn(model, test_loader)

            agg_start_logits += start_logits
            agg_end_logits += end_logits
            models_found += 1

            # Clean up GPU memory
            del model
            del state_dict
            torch.cuda.empty_cache()

        if models_found == 0:
            raise RuntimeError(
                "No model checkpoints found! Cannot generate predictions."
            )

        # Average the logits
        agg_start_logits /= models_found
        agg_end_logits /= models_found

        print("Post-processing aggregated logits...")

        # 5. Post-processing to find best spans
        # Group feature indices by example_id to handle sliding windows
        example_to_indices = features_df.groupby("example_id").indices

        final_predictions = []

        # Iterate over original test examples to ensure output order matches input
        for _, row in test_df.iterrows():
            ex_id = row["id"]
            context_text = row["context"]

            if ex_id not in example_to_indices:
                # Fallback for missing features (unlikely)
                final_predictions.append({"id": ex_id, "PredictionString": ""})
                continue

            feature_indices = example_to_indices[ex_id]

            best_score = -float("inf")
            best_answer = ""

            # Iterate over all sliding window features for this example
            for idx in feature_indices:
                start_logit = agg_start_logits[idx]
                end_logit = agg_end_logits[idx]

                # Retrieve metadata
                offsets = features_df.iloc[idx]["offset_mapping"]
                token_type_ids = features_df.iloc[idx]["token_type_ids"]

                # Ensure numpy format for masking
                if not isinstance(token_type_ids, np.ndarray):
                    token_type_ids = np.array(token_type_ids)

                # Mask non-context tokens
                # MuRIL/BERT convention: 0 = Query, 1 = Context
                context_mask = token_type_ids == 1

                min_score = -1e9
                s_logits = np.where(context_mask, start_logit, min_score)
                e_logits = np.where(context_mask, end_logit, min_score)

                # Get top-k candidates
                start_indexes = np.argsort(s_logits)[-Config.N_BEST_SIZE :][::-1]
                end_indexes = np.argsort(e_logits)[-Config.N_BEST_SIZE :][::-1]

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Basic validity checks
                        if start_index > end_index:
                            continue

                        length = end_index - start_index + 1
                        if length > Config.MAX_ANSWER_LENGTH:
                            continue

                        score = start_logit[start_index] + end_logit[end_index]

                        if score > best_score:
                            best_score = score

                            # Extract text using offsets
                            try:
                                # offsets is likely a list of lists/tuples
                                start_char = int(offsets[start_index][0])
                                end_char = int(offsets[end_index][1])
                                best_answer = context_text[start_char:end_char]
                            except Exception:
                                continue

            final_predictions.append({"id": ex_id, "PredictionString": best_answer})

        # 6. Save Submission
        sub_df = pd.DataFrame(final_predictions)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
