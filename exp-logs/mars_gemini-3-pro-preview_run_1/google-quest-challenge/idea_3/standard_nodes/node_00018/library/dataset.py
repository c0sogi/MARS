import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Define target columns based on the task description
ALL_TARGETS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

# Split targets for decoupled heads
Q_TARGETS = [c for c in ALL_TARGETS if c.startswith("question_")]
A_TARGETS = [c for c in ALL_TARGETS if c.startswith("answer_")]


def preprocess_and_cache(split, load_cached_data=True):
    """
    Loads metadata, preprocesses text features, and caches the result to parquet.
    """
    cache_dir = "./working/idea_3/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # Always read raw input for test set to avoid metadata mismatch
        if split == "test":
            meta_path = "./input/test.csv"
        else:
            meta_path = f"./metadata/{split}.csv"
            if not os.path.exists(meta_path):
                # Fallback for local testing if metadata generation script wasn't run
                # Assuming input files exist in standard location
                meta_path = "./input/train.csv"

        df = pd.read_csv(meta_path)

        # Preprocess text: Handle NaNs and create combined input
        df["question_title"] = df["question_title"].fillna("").astype(str)
        df["question_body"] = df["question_body"].fillna("").astype(str)
        df["answer"] = df["answer"].fillna("").astype(str)

        # Concatenate title and body for question representation
        df["question_input"] = df["question_title"] + " " + df["question_body"]
        df["answer_input"] = df["answer"]

        # Save to cache
        df.to_parquet(cache_path, index=False)

    return df


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract lists for faster access
        self.q_texts = df["question_input"].tolist()
        self.a_texts = df["answer_input"].tolist()
        self.qa_ids = df["qa_id"].tolist()

        if not self.is_test:
            self.q_labels = df[Q_TARGETS].values.astype(np.float32)
            self.a_labels = df[A_TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Tokenize Question (Dynamic Padding: return unpadded lists)
        q_enc = self.tokenizer(
            self.q_texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Tokenize Answer
        a_enc = self.tokenizer(
            self.a_texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "a_input_ids": a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"],
            "qa_id": self.qa_ids[idx],
        }

        if not self.is_test:
            item["q_labels"] = self.q_labels[idx]
            item["a_labels"] = self.a_labels[idx]

        return item


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract lists of inputs
        q_input_ids = [item["q_input_ids"] for item in batch]
        q_masks = [item["q_attention_mask"] for item in batch]
        a_input_ids = [item["a_input_ids"] for item in batch]
        a_masks = [item["a_attention_mask"] for item in batch]
        qa_ids = [item["qa_id"] for item in batch]

        # Helper to pad a list of inputs using tokenizer's pad method
        def pad_sequences(ids, masks):
            # Prepare list of dicts for tokenizer.pad
            features = [
                {"input_ids": i, "attention_mask": m} for i, m in zip(ids, masks)
            ]
            padded = self.tokenizer.pad(
                features,
                padding="longest",  # Pad to max length in this batch
                return_tensors="pt",
            )
            return padded["input_ids"], padded["attention_mask"]

        q_ids_padded, q_masks_padded = pad_sequences(q_input_ids, q_masks)
        a_ids_padded, a_masks_padded = pad_sequences(a_input_ids, a_masks)

        batch_out = {
            "q_input_ids": q_ids_padded,
            "q_attention_mask": q_masks_padded,
            "a_input_ids": a_ids_padded,
            "a_attention_mask": a_masks_padded,
            "qa_id": torch.tensor(qa_ids, dtype=torch.long),
        }

        # Stack labels if present
        if "q_labels" in batch[0]:
            q_labels = torch.stack([torch.tensor(item["q_labels"]) for item in batch])
            a_labels = torch.stack([torch.tensor(item["a_labels"]) for item in batch])
            batch_out["q_labels"] = q_labels
            batch_out["a_labels"] = a_labels

        return batch_out


def get_loaders(
    tokenizer_name="distilroberta-base",
    batch_size=8,
    max_length=512,
    load_cached_data=True,
    num_workers=2,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load processed dataframes
    train_df = preprocess_and_cache("train", load_cached_data)
    val_df = preprocess_and_cache("val", load_cached_data)
    test_df = preprocess_and_cache("test", load_cached_data)

    # Initialize Datasets
    train_ds = QuestDataset(train_df, tokenizer, max_length, is_test=False)
    val_ds = QuestDataset(val_df, tokenizer, max_length, is_test=False)
    test_ds = QuestDataset(test_df, tokenizer, max_length, is_test=True)

    # Initialize Collate function
    collate_fn = Collate(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
