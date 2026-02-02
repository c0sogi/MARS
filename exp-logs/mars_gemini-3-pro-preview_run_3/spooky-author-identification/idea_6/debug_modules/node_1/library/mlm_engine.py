import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_text_data


class MLMDataset(Dataset):
    """
    Dataset wrapper for Masked Language Modeling.
    """

    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.special_tokens_mask = encodings.get("special_tokens_mask", None)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.special_tokens_mask is not None:
            item["special_tokens_mask"] = torch.tensor(
                self.special_tokens_mask[idx], dtype=torch.long
            )
        return item

    def __len__(self):
        return len(self.input_ids)


def sanitize_model_name(model_name):
    """Replaces forward slashes in model names with dashes for file paths."""
    return model_name.replace("/", "-")


def get_tokenized_data(texts, tokenizer, model_name, split_name, load_cached_data=True):
    """
    Tokenizes text data and caches it using Parquet to avoid pickle.

    Args:
        texts (array-like): List of text strings.
        tokenizer: HuggingFace tokenizer.
        model_name (str): Name of the model.
        split_name (str): 'train' or 'val'.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        dict: Dictionary containing input_ids, attention_mask, etc.
    """
    sanitized_name = sanitize_model_name(model_name)
    cache_path = os.path.join(
        Config.WORKING_DIR, f"mlm_tokens_{sanitized_name}_{split_name}.parquet"
    )

    # 1. Try Load
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return {
                "input_ids": df["input_ids"].tolist(),
                "attention_mask": df["attention_mask"].tolist(),
                "special_tokens_mask": (
                    df["special_tokens_mask"].tolist()
                    if "special_tokens_mask" in df.columns
                    else None
                ),
            }
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute
    # Pad to max_length to ensure we can store in a structured way if needed,
    # though lists in parquet handle variable length fine.
    # We use max_length padding here for consistency.
    tokenized = tokenizer(
        list(texts),
        max_length=Config.MAX_LEN,
        truncation=True,
        padding="max_length",
        return_special_tokens_mask=True,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    special_tokens_mask = tokenized["special_tokens_mask"]

    # 3. Save
    # Create DataFrame with lists
    df = pd.DataFrame(
        {
            "input_ids": [list(x) for x in input_ids],
            "attention_mask": [list(x) for x in attention_mask],
            "special_tokens_mask": [list(x) for x in special_tokens_mask],
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "special_tokens_mask": special_tokens_mask,
    }


def run_mlm_pretraining(
    model_backbones=Config.MODEL_BACKBONES, load_cached_data=True, debug=Config.DEBUG
):
    """
    Orchestrates the Domain-Adaptive Pre-training (MLM) process.

    Args:
        model_backbones (list): List of HuggingFace model identifiers.
        load_cached_data (bool): Whether to use cached data/models.
        debug (bool): Whether to run in debug mode (fewer steps/data).

    Returns:
        dict: Mapping of model names to their fine-tuned directory paths.
    """
    seed_everything(Config.SEED)

    # 1. Load Text Data
    # We use Train + Test for adaptation, Val for monitoring.
    train_texts_raw, _, val_texts_raw, _, test_texts_raw, _ = load_text_data(
        load_cached_data=load_cached_data, debug=debug
    )

    mlm_train_texts = np.concatenate([train_texts_raw, test_texts_raw])
    mlm_val_texts = val_texts_raw

    trained_model_paths = {}

    for model_name in model_backbones:
        sanitized_name = sanitize_model_name(model_name)
        output_dir = os.path.join(Config.MLM_MODEL_DIR, f"mlm_{sanitized_name}")

        # Check cache for trained model
        if (
            load_cached_data
            and os.path.exists(output_dir)
            and os.path.exists(os.path.join(output_dir, "config.json"))
        ):
            print(
                f"MLM model for {model_name} found at {output_dir}. Skipping training."
            )
            trained_model_paths[model_name] = output_dir
            continue

        print(f"Starting MLM training for {model_name}...")

        # Initialize Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError:
            # Fallback or retry logic could go here, but we assume internet access or cached models
            print(f"Could not load tokenizer for {model_name}. Skipping.")
            continue

        # Prepare Datasets
        train_encodings = get_tokenized_data(
            mlm_train_texts, tokenizer, model_name, "train", load_cached_data
        )
        val_encodings = get_tokenized_data(
            mlm_val_texts, tokenizer, model_name, "val", load_cached_data
        )

        train_dataset = MLMDataset(train_encodings)
        val_dataset = MLMDataset(val_encodings)

        # Initialize Model
        # Note: For models like DeBERTa-v3, this adds a fresh LM head.
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Training Configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=Config.MLM_EPOCHS,
            per_device_train_batch_size=Config.MLM_BATCH_SIZE,
            per_device_eval_batch_size=Config.MLM_BATCH_SIZE,
            learning_rate=Config.MLM_LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            disable_tqdm=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=Config.SEED,
            dataloader_num_workers=Config.NUM_WORKERS,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_MASK_PROB
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.PATIENCE)],
        )

        # Train
        trainer.train()

        # Evaluate
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", 0.0)
        perplexity = math.exp(eval_loss)

        print(f"[{model_name}] Final Perplexity: {perplexity:.10f}")
        print(f"[{model_name}] Final Eval Loss: {eval_loss:.10f}")

        # Save
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        trained_model_paths[model_name] = output_dir

        # Cleanup
        del model, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()

    return trained_model_paths
