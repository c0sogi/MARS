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
from datasets import Dataset
from library.config import Config
from library.utils import seed_everything


def get_tapt_corpus(load_cached_data=True):
    """
    Aggregates context text from train, val, and test sets for TAPT.
    Implements caching using Parquet.
    """
    cache_path = os.path.join(Config.TAPT_CACHE_DIR, "corpus.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading TAPT corpus from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df["text"].tolist()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Constructing TAPT corpus from metadata...")

    # Load metadata files
    dfs = []
    for path in [Config.TRAIN_META_PATH, Config.VAL_META_PATH, Config.TEST_META_PATH]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: Metadata file not found at {path}")

    if not dfs:
        raise ValueError("No metadata files found to create corpus.")

    full_df = pd.concat(dfs, ignore_index=True)

    # Extract unique contexts to avoid excessive duplication, though some repetition is okay
    # We want the model to learn the domain language.
    # Using 'context' column.
    contexts = full_df["context"].dropna().unique().tolist()

    print(f"Extracted {len(contexts)} unique context paragraphs.")

    # Create DataFrame for caching
    corpus_df = pd.DataFrame({"text": contexts})

    # Save to cache
    os.makedirs(Config.TAPT_CACHE_DIR, exist_ok=True)
    corpus_df.to_parquet(cache_path, index=False)
    print(f"Saved TAPT corpus to cache: {cache_path}")

    return contexts


def run_tapt(
    load_cached_data=True,
    batch_size=Config.TAPT_BATCH_SIZE,
    num_epochs=Config.TAPT_EPOCHS,
    learning_rate=Config.TAPT_LEARNING_RATE,
    seed=42,
):
    """
    Executes the Task-Adaptive Pretraining (TAPT) pipeline.
    Fine-tunes the base model on the domain corpus using MLM.
    """
    seed_everything(seed)

    print("Starting Task-Adaptive Pretraining (TAPT)...")

    # 1. Prepare Data
    texts = get_tapt_corpus(load_cached_data=load_cached_data)

    if not texts:
        raise ValueError("Corpus is empty.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({"text": texts})

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)

    # 3. Preprocessing (Tokenization & Grouping)
    # We tokenize every text, then concatenate them together, then split them in small chunks.
    block_size = Config.MAX_LENGTH

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    print("Tokenizing corpus...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=Config.NUM_WORKERS,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Grouping texts into chunks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=Config.NUM_WORKERS,
        desc="Grouping",
    )

    print(f"Created {len(lm_datasets)} training samples for MLM.")

    # 4. Model
    model = AutoModelForMaskedLM.from_pretrained(Config.BASE_MODEL_NAME)

    # 5. Data Collator
    # This handles the random masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(Config.WORKING_DIR, "tapt_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=(Config.DEVICE == "cuda"),
        disable_tqdm=True,  # Reduce verbosity
        report_to="none",
        logging_strategy="epoch",
        seed=seed,
        data_seed=seed,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 8. Train
    print("Training MLM model...")
    train_result = trainer.train()

    print(
        f"TAPT Training Complete. Global Step: {train_result.global_step}, Training Loss: {train_result.training_loss}"
    )

    # 9. Save Final Model
    print(f"Saving TAPT model to {Config.TAPT_MODEL_DIR}...")
    trainer.save_model(Config.TAPT_MODEL_DIR)
    tokenizer.save_pretrained(Config.TAPT_MODEL_DIR)

    print("TAPT pipeline finished successfully.")
