import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from library.configuration import Config
from library.dataset_factory import prepare_tapt_data
from library.utils import seed_everything


def run_tapt_training(load_cached_data=True):
    """
    Executes Task-Adaptive Pretraining (TAPT) using Masked Language Modeling.

    Fine-tunes xlm-roberta-base on the concatenation of Train, Val, and Test
    context text to adapt the embeddings to the specific Hindi/Tamil domain.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        str: Path to the saved fine-tuned model directory.
    """
    # 1. Reproducibility
    seed_everything(Config.SEEDS[0])

    print(f"Initializing TAPT on device: {Config.DEVICE}")

    # 2. Load Tokenizer
    # We need the tokenizer to prepare the dataset and for the data collator
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # 3. Prepare Data
    # This loads text from all splits, tokenizes, chunks, and caches it.
    train_dataset = prepare_tapt_data(tokenizer, load_cached_data=load_cached_data)

    print(f"TAPT Dataset loaded. Number of samples: {len(train_dataset)}")

    # 4. Initialize Model
    # Load the base Masked LM model
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_CHECKPOINT)
    model.to(Config.DEVICE)

    # 5. Data Collator
    # Handles dynamic masking of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.TAPT_CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        learning_rate=Config.TAPT_LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        save_strategy="no",  # We only care about the final model for initialization
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=True,
        seed=Config.SEEDS[0],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=Config.NUM_WORKERS,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 8. Train
    print("Starting TAPT training...")
    train_result = trainer.train()

    print(f"TAPT training complete. Final Training Loss: {train_result.training_loss}")

    # 9. Save Final Model
    # This model will be used as the starting point for QA fine-tuning
    print(f"Saving adapted model to {Config.TAPT_OUTPUT_DIR}...")
    trainer.save_model(Config.TAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.TAPT_OUTPUT_DIR)

    return Config.TAPT_OUTPUT_DIR
