import os
import math
import torch
from torch.utils.data import random_split
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from library.configuration import Config, seed_everything
from library.utilities import get_logger
from library.dataset import get_tokenizer, load_mlm_data

logger = get_logger("Pretraining")


def run_pretraining(debug: bool = False, load_cached_data: bool = True):
    """
    Executes the Domain-Adaptive Pre-training (DAPT) stage using Masked Language Modeling.

    Args:
        debug (bool): If True, runs on a small subset of data for testing.
        load_cached_data (bool): If True, attempts to load processed data from cache.
    """
    seed_everything(Config.SEED)

    logger.info("Initializing MLM Pre-training...")

    # 1. Prepare Data
    tokenizer = get_tokenizer()
    full_dataset = load_mlm_data(
        tokenizer, load_cached_data=load_cached_data, debug=debug
    )

    # Create a validation split for Early Stopping (10% of data)
    # Even in unsupervised learning, holding out data helps monitor overfitting (perplexity)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    logger.info(
        f"Data loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}"
    )

    # 2. Initialize Model
    # We use AutoModelForMaskedLM to get the pre-trained backbone + MLM head
    logger.info(f"Loading model: {Config.MODEL_BACKBONE}")
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_BACKBONE)

    if Config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    # 3. Setup Training Components
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_MASK_PROB
    )

    # Infer gradient accumulation to match effective batch size of supervised stage (~16)
    # MLM_BATCH_SIZE is 2, so 8 steps accumulation
    grad_accum_steps = 8

    training_args = TrainingArguments(
        output_dir=Config.MLM_CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.MLM_EPOCHS,
        per_device_train_batch_size=Config.MLM_BATCH_SIZE,
        per_device_eval_batch_size=Config.MLM_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=Config.MLM_LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        fp16=torch.cuda.is_available(),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="none",  # Disable wandb/mlflow
        dataloader_num_workers=Config.NUM_WORKERS,
        seed=Config.SEED,
        data_seed=Config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 4. Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # 5. Log Metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    eval_metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"Final Validation Loss: {eval_metrics['eval_loss']}")
    logger.info(f"Final Perplexity: {perplexity}")

    # 6. Save Final Model
    logger.info(f"Saving adapted model to {Config.MLM_CHECKPOINT_DIR}")
    trainer.save_model(Config.MLM_CHECKPOINT_DIR)
    tokenizer.save_pretrained(Config.MLM_CHECKPOINT_DIR)

    logger.info("MLM Pre-training completed successfully.")
