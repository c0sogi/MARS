import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import numpy as np
import pandas as pd

from library.config import (
    MODEL_TYPE,
    MAX_LEN,
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
    CACHE_DIR,
)
from library.utils import jaccard, set_seed


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_TYPE)
        self.drop = nn.Dropout(0.1)
        self.l0 = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # outputs[0] is sequence_output: (batch_size, seq_len, hidden_size)
        out = self.drop(outputs[0])
        logits = self.l0(out)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        text = " " + " ".join(str(row.text).split())
        sentiment = str(row.sentiment)

        # Encode: sentiment [SEP] text
        # We use encode_plus to handle special tokens and offsets
        encoded = self.tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]

        # Targets
        target_idx_start = 0
        target_idx_end = 0

        if "selected_text" in row:
            selected_text = " " + " ".join(str(row.selected_text).split())
            start_char = text.find(selected_text)

            if start_char != -1:
                end_char = start_char + len(selected_text)

                sequence_ids = encoded.sequence_ids()
                # Find tokens corresponding to text (sequence_id == 1)
                text_token_indices = [
                    i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
                ]

                if text_token_indices:
                    token_start_index = text_token_indices[0]
                    token_end_index = text_token_indices[-1]

                    # Find start token
                    for i in range(token_start_index, token_end_index + 1):
                        if offsets[i][0] <= start_char and offsets[i][1] > start_char:
                            target_idx_start = i
                            break
                        elif (
                            offsets[i][0] >= start_char
                        ):  # Fallback if exact overlap missed
                            target_idx_start = i
                            break

                    # Find end token
                    for i in range(token_end_index, token_start_index - 1, -1):
                        if offsets[i][0] < end_char and offsets[i][1] >= end_char:
                            target_idx_end = i
                            break
                        elif offsets[i][1] <= end_char:  # Fallback
                            target_idx_end = i
                            break

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets_start": torch.tensor(target_idx_start, dtype=torch.long),
            "targets_end": torch.tensor(target_idx_end, dtype=torch.long),
            "text": text,
            "sentiment": sentiment,
            "offsets": torch.tensor(offsets, dtype=torch.long),
        }


class SentimentRelevanceModel:
    """
    RoBERTa-based Span Extraction Model.
    Cite solution_lesson_node_00004: Contextual Span Prediction vs. Statistical Token Weighting.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
        self.model = TweetModel()
        self.model.to(self.device)
        set_seed(SEED)

    def fit(self, train_df, load_cached_data=True):
        # Check cache
        if load_cached_data and os.path.exists(MODEL_SAVE_PATH):
            print(f"Loading model from {MODEL_SAVE_PATH}...")
            self.model.load_state_dict(
                torch.load(MODEL_SAVE_PATH, map_location=self.device)
            )
            return

        # Cite solution_lesson_node_00006: Stratified Training with Deterministic Inference Rules
        # Exclude neutral tweets from training as they are handled by a deterministic rule
        print("Filtering out neutral tweets for training...")
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)

        print("Training RoBERTa model...")
        train_dataset = TweetDataset(train_df, self.tokenizer, MAX_LEN)
        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
        )

        optimizer = AdamW(self.model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(EPOCHS):
            # Simple training loop
            total_loss = 0
            for data in train_loader:
                ids = data["ids"].to(self.device)
                mask = data["mask"].to(self.device)
                start_targets = data["targets_start"].to(self.device)
                end_targets = data["targets_end"].to(self.device)

                optimizer.zero_grad()
                start_logits, end_logits = self.model(ids, mask)

                loss = loss_fn(start_logits, start_targets) + loss_fn(
                    end_logits, end_targets
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

        # Save model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved.")

    def predict(self, test_df):
        self.model.eval()
        test_dataset = TweetDataset(test_df, self.tokenizer, MAX_LEN)
        test_loader = DataLoader(
            test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2
        )

        predictions = []

        with torch.no_grad():
            for data in test_loader:
                ids = data["ids"].to(self.device)
                mask = data["mask"].to(self.device)
                offsets = data["offsets"].cpu().numpy()
                original_texts = data["text"]
                sentiments = data["sentiment"]

                start_logits, end_logits = self.model(ids, mask)
                start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                for i in range(len(ids)):
                    sentiment = sentiments[i]
                    text = original_texts[i]

                    # Cite solution_lesson_node_00001: Neutral tweets have high overlap -> predict full text.
                    if sentiment == "neutral" or len(text.split()) < 2:
                        predictions.append(text.strip())
                        continue

                    start_idx = np.argmax(start_probs[i])
                    end_idx = np.argmax(end_probs[i])

                    if start_idx > end_idx:
                        end_idx = start_idx

                    _offsets = offsets[i]

                    # Decode
                    # Get char offsets
                    if start_idx < len(_offsets) and end_idx < len(_offsets):
                        start_char = _offsets[start_idx][0]
                        end_char = _offsets[end_idx][1]
                        pred_text = text[start_char:end_char]
                    else:
                        pred_text = text

                    predictions.append(pred_text.strip())

        return pd.DataFrame({"textID": test_df["textID"], "selected_text": predictions})

    def evaluate(self, val_df):
        print("Evaluating model...")
        preds_df = self.predict(val_df)
        scores = []
        for i in range(len(val_df)):
            target = str(val_df.iloc[i]["selected_text"])
            pred = str(preds_df.iloc[i]["selected_text"])
            scores.append(jaccard(target, pred))

        mean_score = np.mean(scores)
        print(f"Validation Jaccard Score: {mean_score}")
        return mean_score

    def generate_submission(self, test_df):
        print("Generating submission...")
        preds_df = self.predict(test_df)
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        preds_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")


def run_model_pipeline(debug=False, debug_size=500):
    """
    Helper function to run the full pipeline: Load -> Fit -> Evaluate -> Submit.
    """
    # 1. Load Data
    train_df, val_df, test_df = load_processed_data(
        load_cached_data=True, debug=debug, debug_size=debug_size
    )

    # 2. Initialize and Fit Model
    model = SentimentRelevanceModel()
    model.fit(train_df, load_cached_data=True)

    # 3. Evaluate
    model.evaluate(val_df)

    # 4. Generate Submission
    model.generate_submission(test_df)
