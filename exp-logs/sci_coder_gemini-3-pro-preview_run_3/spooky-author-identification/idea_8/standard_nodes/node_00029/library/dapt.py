import os
import pandas as pd
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from library.config import PathConfig, ModelConfig, TrainConfig
from library.utils import set_seed
from library.data_processing import load_data, MLMDataset


def get_mlm_corpus(load_cached_data=True):
    """
    Prepares the corpus for Masked Language Modeling by combining train, val, and test texts.
    Implements caching using parquet to ensure deterministic processing.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        list: A list of text strings.
    """
    cache_file = os.path.join(PathConfig.WORKING_DIR, "mlm_corpus.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached MLM corpus from {cache_file}...")
        try:
            df = pd.read_parquet(cache_file)
            return df["text"].tolist()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Generating MLM corpus from source files...")
    train_df, val_df, test_df = load_data()

    # Concatenate all text to maximize domain adaptation data
    all_texts = pd.concat(
        [train_df["text"], val_df["text"], test_df["text"]], axis=0
    ).reset_index(drop=True)

    # Save to cache
    os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)
    pd.DataFrame({"text": all_texts}).to_parquet(cache_file)
    print(f"Saved MLM corpus to {cache_file}")

    return all_texts.tolist()


def run_mlm_pretraining(debug=False, load_cached_data=True):
    """
    Runs Domain-Adaptive Pre-training (DAPT) using MLM on the provided backbones.

    Args:
        debug (bool): If True, runs on a small subset of data for 1 epoch.
        load_cached_data (bool): If True, attempts to load processed corpus from cache.
    """
    set_seed(TrainConfig.SEED)

    # Ensure directories exist
    os.makedirs(PathConfig.MLM_MODELS_DIR, exist_ok=True)

    # Get Corpus
    texts = get_mlm_corpus(load_cached_data=load_cached_data)

    if debug:
        print("DEBUG MODE: Truncating corpus to 100 samples.")
        texts = texts[:100]
        epochs = 1
    else:
        epochs = TrainConfig.DAPT_EPOCHS

    print(f"MLM Corpus Size: {len(texts)} samples")

    for backbone in ModelConfig.BACKBONES:
        print(f"\n{'='*40}")
        print(f"Processing Backbone: {backbone}")
        print(f"{'='*40}")

        # Create a safe directory name for the model
        model_dir_name = backbone.replace("/", "-")
        output_dir = os.path.join(PathConfig.MLM_MODELS_DIR, f"mlm_{model_dir_name}")

        # Check if model already exists to avoid retraining
        # We check for config.json and model weights
        is_trained = os.path.exists(os.path.join(output_dir, "config.json")) and (
            os.path.exists(os.path.join(output_dir, "model.safetensors"))
            or os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
        )

        if is_trained and load_cached_data:
            print(f"Model already found at {output_dir}. Skipping training.")
            continue

        print(f"Training MLM model for {backbone}...")

        # Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(backbone)
        except Exception as e:
            print(f"Error loading tokenizer for {backbone}: {e}")
            continue

        # Prepare Dataset
        # MLMDataset expects list of strings and handles tokenization
        dataset = MLMDataset(
            texts=texts, tokenizer=tokenizer, max_length=ModelConfig.MAX_LENGTH
        )

        # Data Collator for MLM (handles masking)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=TrainConfig.DAPT_MASK_PROB,
        )

        # Load Model
        model = AutoModelForMaskedLM.from_pretrained(backbone)

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=TrainConfig.DAPT_BATCH_SIZE,
            learning_rate=TrainConfig.DAPT_LR,
            weight_decay=TrainConfig.WEIGHT_DECAY,
            save_strategy="no",  # Save only at the end to save space
            logging_strategy="steps",
            logging_steps=50,
            seed=TrainConfig.SEED,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=TrainConfig.NUM_WORKERS,
            disable_tqdm=True,  # Silent execution
            report_to="none",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Train
        train_result = trainer.train()

        # Print metrics
        print(f"Training completed for {backbone}.")
        print(f"Training Loss: {train_result.training_loss}")

        # Save Model and Tokenizer
        print(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    print("\nAll MLM pre-training tasks completed.")
