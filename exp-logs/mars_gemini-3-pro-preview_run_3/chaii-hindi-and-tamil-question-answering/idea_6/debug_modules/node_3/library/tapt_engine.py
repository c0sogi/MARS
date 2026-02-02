import os
import math
import pandas as pd
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

from library.configuration import Config
from library.utilities import set_seed


def prepare_tapt_data(load_cached_data=True):
    """
    Prepares the text corpus for Task-Adaptive Pretraining.
    Extracts contexts from Train, Val, and Test metadata.
    Caches the result as parquet files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_dataset_path, val_dataset_path)
    """
    # Ensure cache directory exists
    os.makedirs(Config.TAPT_CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.TAPT_CACHE_DIR, "train_corpus.parquet")
    val_cache_path = os.path.join(Config.TAPT_CACHE_DIR, "val_corpus.parquet")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
    ):
        print(f"Loading cached TAPT corpus from {Config.TAPT_CACHE_DIR}")
        return train_cache_path, val_cache_path

    # 2. Process from scratch
    print("Preparing TAPT corpus from metadata...")

    # Load metadata
    try:
        df_train = pd.read_csv(Config.TRAIN_FILE)
        df_val = pd.read_csv(Config.VAL_FILE)
        df_test = pd.read_csv(Config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"Error loading metadata files: {e}")
        raise

    # Apply debugging limits if specified in Config
    if Config.MAX_TRAIN_SAMPLES:
        df_train = df_train.iloc[: Config.MAX_TRAIN_SAMPLES]
        df_test = df_test.iloc[: Config.MAX_TRAIN_SAMPLES]
    if Config.MAX_VAL_SAMPLES:
        df_val = df_val.iloc[: Config.MAX_VAL_SAMPLES]

    # Extract unique contexts
    # Combine Train and Test contexts for the MLM training set (unsupervised domain adaptation)
    # Use Val contexts for the MLM validation set to monitor perplexity
    train_contexts = pd.concat(
        [df_train["context"], df_test["context"]], ignore_index=True
    ).unique()
    val_contexts = df_val["context"].unique()

    # Create DataFrames for parquet storage
    df_corpus_train = pd.DataFrame({"text": train_contexts})
    df_corpus_val = pd.DataFrame({"text": val_contexts})

    # Save to parquet
    df_corpus_train.to_parquet(train_cache_path, index=False)
    df_corpus_val.to_parquet(val_cache_path, index=False)

    print(f"Saved TAPT corpus to {Config.TAPT_CACHE_DIR}")

    return train_cache_path, val_cache_path


def run_tapt_pretraining(load_cached_data=True):
    """
    Runs the Task-Adaptive Pretraining (MLM) on the domain data.

    Args:
        load_cached_data (bool): Whether to use cached data preparation.
    """
    set_seed(Config.SEED)

    # 1. Prepare Data
    train_path, val_path = prepare_tapt_data(load_cached_data=load_cached_data)

    # Load datasets using HuggingFace datasets from the parquet files
    dataset_train = Dataset.from_parquet(train_path)
    dataset_val = Dataset.from_parquet(val_path)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # 3. Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_special_tokens_mask=True,
        )

    print("Tokenizing TAPT datasets...")
    # Use multiprocessing for faster tokenization
    tokenized_train = dataset_train.map(
        tokenize_function,
        batched=True,
        num_proc=Config.NUM_WORKERS,
        remove_columns=["text"],
    )
    tokenized_val = dataset_val.map(
        tokenize_function,
        batched=True,
        num_proc=Config.NUM_WORKERS,
        remove_columns=["text"],
    )

    # 4. Data Collator
    # Handles dynamic masking for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.TAPT_MLM_PROBABILITY
    )

    # 5. Model
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_CHECKPOINT)
    model.to(Config.DEVICE)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.TAPT_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        per_device_eval_batch_size=Config.TAPT_BATCH_SIZE * 2,
        learning_rate=Config.TAPT_LEARNING_RATE,
        weight_decay=Config.TAPT_WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=Config.SEED,
        data_seed=Config.SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=Config.NUM_WORKERS,
        report_to="none",
        disable_tqdm=True,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 8. Train
    print("Starting TAPT training...")
    train_result = trainer.train()

    # Print metrics
    print(f"Training metrics: {train_result.metrics}")

    # Evaluate
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")

    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    print(f"Final Validation Perplexity: {perplexity}")

    # 9. Save adapted model
    print(f"Saving TAPT model to {Config.TAPT_OUTPUT_DIR}")
    trainer.save_model(Config.TAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.TAPT_OUTPUT_DIR)
