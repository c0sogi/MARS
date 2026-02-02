import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import set_seed


class MlmDataset(Dataset):
    """
    Custom Dataset for Masked Language Modeling.
    Reads a text file line-by-line and tokenizes it.
    """

    def __init__(self, tokenizer, lines, max_length):
        self.tokenizer = tokenizer
        self.examples = []

        # Tokenize all lines efficiently
        # We use batch_encode_plus for speed, though doing it in __getitem__
        # is more memory efficient for huge datasets. Given the dataset size
        # here (~1000 docs), pre-tokenization is fine and faster for training.
        batch_encoding = tokenizer(
            lines,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        self.input_ids = batch_encoding["input_ids"]
        self.attention_mask = batch_encoding["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            # labels are handled by the DataCollator
        }


def create_mlm_corpus(load_cached_data=True):
    """
    Creates a text corpus from train, val, and test contexts for MLM.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        str: Path to the generated corpus file.
    """
    corpus_path = os.path.join(Config.TAPT_CACHE_DIR, "corpus.txt")

    # Check cache
    if load_cached_data and os.path.exists(corpus_path):
        print(f"Loading cached MLM corpus from {corpus_path}")
        return corpus_path

    print("Generating MLM corpus from metadata...")

    # Load all metadata files
    dfs = []
    for path in [Config.TRAIN_META_PATH, Config.VAL_META_PATH, Config.TEST_META_PATH]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise ValueError("No metadata files found to create corpus.")

    full_df = pd.concat(dfs, ignore_index=True)

    # Extract unique contexts
    # We drop duplicates to avoid bias if contexts are repeated
    contexts = full_df["context"].dropna().unique().tolist()

    # Clean text (basic whitespace normalization)
    contexts = [c.replace("\n", " ").strip() for c in contexts if len(c.strip()) > 0]

    print(f"Extracted {len(contexts)} unique contexts.")

    # Save to file
    with open(corpus_path, "w", encoding="utf-8") as f:
        for ctx in contexts:
            f.write(ctx + "\n")

    print(f"Saved corpus to {corpus_path}")
    return corpus_path


def run_tapt():
    """
    Runs Task-Adaptive Pretraining (TAPT) using Masked Language Modeling.
    Fine-tunes xlm-roberta-base on the domain text.
    """
    set_seed(42)

    # 1. Prepare Data
    corpus_path = create_mlm_corpus(load_cached_data=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Split into train and validation for early stopping monitoring
    # Even in TAPT, it's good to ensure we aren't overfitting the specific texts too hard
    train_lines, val_lines = train_test_split(lines, test_size=0.1, random_state=42)

    print(
        f"TAPT Data: {len(train_lines)} training lines, {len(val_lines)} validation lines."
    )

    # 2. Setup Model and Tokenizer
    print(f"Loading base model: {Config.MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_CHECKPOINT)

    # 3. Create Datasets
    train_dataset = MlmDataset(tokenizer, train_lines, Config.MAX_LENGTH)
    val_dataset = MlmDataset(tokenizer, val_lines, Config.MAX_LENGTH)

    # 4. Data Collator (Handles Masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.TAPT_MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        per_device_eval_batch_size=Config.TAPT_BATCH_SIZE,
        learning_rate=Config.TAPT_LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard for this script
        disable_tqdm=True,  # Cleaner output
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train
    print("Starting TAPT training...")
    train_result = trainer.train()

    # Print metrics
    print("Training metrics:")
    for key, value in train_result.metrics.items():
        print(f"{key}: {value}")

    eval_metrics = trainer.evaluate()
    print("Final Evaluation metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    # 8. Save Final Adapted Model
    print(f"Saving TAPT model to {Config.TAPT_MODEL_DIR}")
    trainer.save_model(Config.TAPT_MODEL_DIR)
    tokenizer.save_pretrained(Config.TAPT_MODEL_DIR)

    print("TAPT complete.")
