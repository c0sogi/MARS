import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from library.config import (
    MODEL_NAME,
    MAX_LEN,
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
    CACHE_DIR,
)
from library.utils import jaccard, set_seed


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = df["text"].values
        self.sentiment = df["sentiment"].values
        self.selected_text = (
            df["selected_text"].values if "selected_text" in df.columns else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = " " + " ".join(str(self.text[item]).split())
        sentiment = self.sentiment[item]

        # Tokenize: <s> sentiment </s> </s> text </s>
        # We use encode_plus to handle the special tokens automatically
        # But we need offset mapping for the text part.

        encoding = self.tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        ids = encoding["input_ids"]
        mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]

        targets_start = 0
        targets_end = 0

        if self.selected_text is not None:
            selected_text = " " + " ".join(str(self.selected_text[item]).split())

            # Find char start/end in the cleaned text
            start_idx = text.find(selected_text)
            end_idx = start_idx + len(selected_text)

            chars = [0] * len(text)
            if start_idx != -1:
                for c in range(start_idx, end_idx):
                    chars[c] = 1

            # Map chars to tokens
            token_labels = []
            idx = 0
            for start, end in offsets:
                if start == end:  # Special token
                    token_labels.append(0)
                elif (
                    encoding.sequence_ids(0)[idx] == 1
                ):  # This token belongs to sequence 1 (text)
                    # Check overlap
                    if sum(chars[start:end]) > 0:
                        token_labels.append(1)
                    else:
                        token_labels.append(0)
                else:
                    token_labels.append(0)
                idx += 1

            # Find first and last 1
            if 1 in token_labels:
                targets_start = token_labels.index(1)
                targets_end = len(token_labels) - 1 - token_labels[::-1].index(1)
            else:
                targets_start = 0
                targets_end = 0

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "text": text,
            "sentiment": sentiment,
            "offsets": torch.tensor(offsets, dtype=torch.long),
        }


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        config = RobertaConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME, config=config)
        self.dropout = nn.Dropout(0.15)
        self.l0 = nn.Linear(config.hidden_size, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask):
        out = self.roberta(ids, attention_mask=mask)
        sequence_output = out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


class SentimentRelevanceModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(SEED)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
        self.model = TweetModel()
        self.model.to(self.device)

    def fit(self, train_df, load_cached_data=True):
        if load_cached_data and os.path.exists(MODEL_SAVE_PATH):
            print("Loading model from cache...")
            self.model.load_state_dict(
                torch.load(MODEL_SAVE_PATH, map_location=self.device)
            )
            return

        print("Training model...")
        # Train only on non-neutral data
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)

        # Filter out rows where selected_text is not found in text after normalization
        # This prevents training on (0,0) targets caused by preprocessing mismatches
        # Cite solution_lesson_node_00012: Robustness of Mask-Based Span Mapping
        def has_valid_span(row):
            text = " " + " ".join(str(row["text"]).split())
            selected_text = " " + " ".join(str(row["selected_text"]).split())
            return text.find(selected_text) != -1

        initial_len = len(train_df)
        train_df = train_df[train_df.apply(has_valid_span, axis=1)].reset_index(
            drop=True
        )
        print(f"Filtered {initial_len - len(train_df)} rows with invalid spans.")

        train_dataset = TweetDataset(train_df, self.tokenizer, MAX_LEN)
        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        # Cite solution_lesson_node_00007: Constrain model capacity and training.
        # We keep EPOCHS=3 and use a scheduler to optimize convergence within this short window.
        num_train_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        # Use Label Smoothing to handle noisy span boundaries in subjective sentiment data
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for data in train_loader:
                ids = data["ids"].to(self.device)
                mask = data["mask"].to(self.device)
                ts = data["targets_start"].to(self.device)
                te = data["targets_end"].to(self.device)

                optimizer.zero_grad()
                o1, o2 = self.model(ids, mask)

                loss = criterion(o1, ts) + criterion(o2, te)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved.")

    def predict(self, test_df):
        self.model.eval()
        predictions = []

        test_dataset = TweetDataset(test_df, self.tokenizer, MAX_LEN)
        test_loader = DataLoader(
            test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2
        )

        with torch.no_grad():
            for data in test_loader:
                ids = data["ids"].to(self.device)
                mask = data["mask"].to(self.device)
                offsets = data["offsets"].numpy()
                texts = data["text"]
                sentiments = data["sentiment"]

                o1, o2 = self.model(ids, mask)
                o1 = torch.softmax(o1, dim=1).cpu().numpy()
                o2 = torch.softmax(o2, dim=1).cpu().numpy()

                for i in range(len(ids)):
                    sentiment = sentiments[i]
                    text = texts[i]

                    if sentiment == "neutral":
                        predictions.append(text)
                    else:
                        # Joint Probability Maximization Decoding
                        # Find (s, e) such that s <= e maximizing P(s) * P(e)
                        p_start = o1[i]
                        p_end = o2[i]

                        # Outer product to get joint matrix
                        scores = np.outer(p_start, p_end)

                        # Mask out invalid spans (where start > end)
                        # np.triu returns upper triangle (including diagonal)
                        scores = np.triu(scores)

                        # Find global maximum
                        flat_idx = np.argmax(scores)
                        start_idx, end_idx = np.unravel_index(flat_idx, scores.shape)

                        try:
                            char_start = offsets[i][start_idx][0]
                            char_end = offsets[i][end_idx][1]

                            if (
                                char_start == 0
                                and char_end == 0
                                and start_idx != end_idx
                            ):
                                pred_text = text
                            else:
                                pred_text = text[char_start:char_end]
                        except:
                            pred_text = text

                        predictions.append(pred_text)

        return pd.DataFrame({"textID": test_df["textID"], "selected_text": predictions})

    def evaluate(self, val_df):
        print("Evaluating model on validation set...")
        preds_df = self.predict(val_df)

        scores = []
        for i in range(len(val_df)):
            target = val_df.iloc[i]["selected_text"]
            pred = preds_df.iloc[i]["selected_text"]
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
