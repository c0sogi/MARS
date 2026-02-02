import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import seed_everything


class MLMDataset(Dataset):
    """
    Simple PyTorch Dataset for Masked Language Modeling.
    Wraps pre-tokenized input_ids and attention_mask.
    """

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }


def load_and_prepare_corpus(debug=False):
    """
    Loads metadata files and concatenates text columns to form a raw corpus.
    """
    paths = [Config.TRAIN_META_PATH, Config.VAL_META_PATH, Config.TEST_META_PATH]
    texts = []

    print("[DAPT] Loading metadata files for corpus generation...")
    for p in paths:
        if not os.path.exists(p):
            print(f"[DAPT] Warning: Metadata file {p} not found. Skipping.")
            continue

        df = pd.read_csv(p)
        if debug:
            df = df.head(100)

        # Fill NaNs and concatenate
        # Format: Title + " " + Body + " " + Answer
        t = (
            df["question_title"].fillna("").astype(str)
            + " "
            + df["question_body"].fillna("").astype(str)
            + " "
            + df["answer"].fillna("").astype(str)
        )

        texts.extend(t.tolist())

    print(f"[DAPT] Total corpus size: {len(texts)} documents.")
    return texts


def get_tokenized_data(tokenizer, load_cached_data=True, debug=False):
    """
    Handles tokenization with caching using .npy files.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, "mlm_data_input_ids.npy")
    mask_path = os.path.join(cache_dir, "mlm_data_attention_mask.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(ids_path) and os.path.exists(mask_path):
        print(f"[DAPT] Loading tokenized data from cache: {cache_dir}")
        input_ids = np.load(ids_path)
        attention_mask = np.load(mask_path)

        if debug:
            print("[DAPT] Debug mode: Subsetting cached data.")
            input_ids = input_ids[:100]
            attention_mask = attention_mask[:100]

        return input_ids, attention_mask

    # 2. Process from Scratch
    print("[DAPT] Cache not found or disabled. Processing raw corpus...")
    texts = load_and_prepare_corpus(debug=debug)

    print(f"[DAPT] Tokenizing {len(texts)} documents (Max Len: {Config.MAX_LEN})...")
    encoding = tokenizer(
        texts,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_attention_mask=True,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # 3. Save to Cache
    print(f"[DAPT] Saving tokenized data to {cache_dir}...")
    np.save(ids_path, input_ids)
    np.save(mask_path, attention_mask)

    return input_ids, attention_mask


def run_dapt(load_cached_data=True, debug=False, epochs=None):
    """
    Main execution function for Domain-Adaptive Pre-Training.

    Args:
        load_cached_data (bool): Whether to use cached tokenized data.
        debug (bool): If True, runs on a small subset for testing.
        epochs (int, optional): Override default epoch count.
    """
    seed_everything(Config.SEED)

    # Setup Paths
    output_dir = os.path.join(Config.WORKING_DIR, "dapt_checkpoints")
    final_model_dir = os.path.join(Config.WORKING_DIR, "dapt_demo_model")

    # Use Config epochs if not provided
    if epochs is None:
        epochs = Config.EPOCHS

    print("=" * 40)
    print(" STARTING DOMAIN ADAPTIVE PRE-TRAINING")
    print("=" * 40)
    print(f"Model: {Config.MLM_MODEL_NAME}")
    print(f"Epochs: {epochs}")
    print(f"Debug Mode: {debug}")

    # 1. Initialize Tokenizer
    # We use the base model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MLM_MODEL_NAME, use_fast=True)

    # 2. Prepare Data
    input_ids, attention_mask = get_tokenized_data(tokenizer, load_cached_data, debug)

    # Split into Train/Val for monitoring (90/10 split)
    train_ids, val_ids, train_mask, val_mask = train_test_split(
        input_ids, attention_mask, test_size=0.1, random_state=Config.SEED
    )

    train_dataset = MLMDataset(train_ids, train_mask)
    val_dataset = MLMDataset(val_ids, val_mask)

    print(f"[DAPT] Train Set: {len(train_dataset)}, Val Set: {len(val_dataset)}")

    # 3. Initialize Model
    # AutoModelForMaskedLM will instantiate a head for MLM.
    # For DeBERTa-v3, this creates a new linear head on top of the encoder.
    model = AutoModelForMaskedLM.from_pretrained(Config.MLM_MODEL_NAME)

    # 4. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.VALID_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION_STEPS,
        learning_rate=2e-5,  # Typical DAPT learning rate
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=(Config.DEVICE == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        report_to="none",  # Disable wandb/mlflow
        disable_tqdm=True,  # Silent execution requirement
        seed=Config.SEED,
        data_seed=Config.SEED,
    )

    # Data Collator handles the random masking of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 6. Train
    print("[DAPT] Starting training...")
    train_result = trainer.train()

    # Log metrics manually since tqdm is disabled
    print(f"[DAPT] Training completed. Global Step: {train_result.global_step}")
    print(f"[DAPT] Train Loss: {train_result.training_loss:.6f}")

    metrics = trainer.evaluate()
    print(f"[DAPT] Final Validation Loss: {metrics['eval_loss']:.6f}")
    print(f"[DAPT] Final Validation Perplexity: {np.exp(metrics['eval_loss']):.6f}")

    # 7. Save Final Adapted Model
    print(f"[DAPT] Saving adapted model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print("[DAPT] Process finished successfully.")
    return final_model_dir
