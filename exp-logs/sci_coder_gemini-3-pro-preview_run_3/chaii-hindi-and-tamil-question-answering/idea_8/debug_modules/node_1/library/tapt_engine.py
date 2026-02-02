import os
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
from library.config import Config
from library.utils import set_seed


def prepare_mlm_data(config: Config, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Prepares the corpus for Masked Language Modeling (TAPT).
    Extracts contexts from Train, Val, and Test metadata to form a domain-specific corpus.
    """
    cache_path = os.path.join(config.tapt_cache_dir, "corpus.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached TAPT corpus from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating TAPT corpus from metadata...")
    # 2. Load Metadata
    # We aggregate context from all available splits to maximize domain exposure
    dfs = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(config.metadata_dir, split)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise ValueError(f"No metadata files found in {config.metadata_dir}")

    full_df = pd.concat(dfs, ignore_index=True)

    # 3. Extract and Clean Contexts
    # We use unique contexts to prevent bias towards frequently repeated paragraphs
    contexts = full_df["context"].dropna().unique()
    corpus_df = pd.DataFrame({"text": contexts})

    # 4. Save Cache
    os.makedirs(config.tapt_cache_dir, exist_ok=True)
    corpus_df.to_parquet(cache_path, index=False)

    print(f"TAPT corpus generated with {len(corpus_df)} unique documents.")
    return corpus_df


def run_tapt(config: Config, load_cached_data: bool = True):
    """
    Runs Task-Adaptive Pretraining (TAPT) using Masked Language Modeling.
    Fine-tunes the base model on the domain corpus to adapt embeddings.
    """
    set_seed(config.seed)
    tapt_config = config.get_tapt_config()

    # 1. Prepare Data
    df = prepare_mlm_data(config, load_cached_data=load_cached_data)

    # Shuffle and Split (90/10) for Loss Monitoring
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    val_size = max(1, int(0.1 * len(df)))

    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]

    print(f"TAPT Data Split - Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(val_df)

    # 2. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 3. Preprocessing: Tokenize and Group
    # We group texts into blocks of 512 to handle long contexts efficiently
    block_size = 512

    def tokenize_function(examples):
        # Do not truncate here; we handle length in group_texts
        return tokenizer(
            examples["text"], return_special_tokens_mask=True, truncation=False
        )

    print("Tokenizing TAPT dataset...")
    # Map with num_proc for speed
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=config.num_workers,
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=config.num_workers,
    )

    def group_texts(examples):
        # Concatenate all texts in the batch
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the small remainder at the end of the batch
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split into chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Grouping texts into blocks...")
    lm_train = tokenized_train.map(
        group_texts, batched=True, num_proc=config.num_workers
    )
    lm_eval = tokenized_eval.map(group_texts, batched=True, num_proc=config.num_workers)

    # 4. Model Setup
    model = AutoModelForMaskedLM.from_pretrained(config.model_name)

    # 5. Training Configuration
    training_args = TrainingArguments(
        output_dir=tapt_config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=tapt_config["num_train_epochs"],
        per_device_train_batch_size=tapt_config["train_batch_size"],
        per_device_eval_batch_size=tapt_config["train_batch_size"],
        learning_rate=tapt_config["learning_rate"],
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        seed=config.seed,
        disable_tqdm=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=tapt_config["mlm_probability"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Execute Training
    print("Starting TAPT Training...")
    trainer.train()

    # 7. Save Artifacts
    print(f"Saving TAPT model to {tapt_config['output_dir']}...")
    trainer.save_model(tapt_config["output_dir"])
    tokenizer.save_pretrained(tapt_config["output_dir"])
    print("TAPT process completed successfully.")
