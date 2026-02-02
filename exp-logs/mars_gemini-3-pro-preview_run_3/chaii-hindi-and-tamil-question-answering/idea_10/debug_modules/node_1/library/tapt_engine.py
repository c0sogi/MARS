import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from library.config import Config
from library.utils import set_seed
from library.data_processing import prepare_tapt_data, get_tokenizer


def run_tapt_training(force_retrain=False):
    """
    Executes the Task-Adaptive Pretraining (TAPT) stage.
    Fine-tunes the base model on the domain dataset using Masked Language Modeling.

    Args:
        force_retrain (bool): If True, forces retraining even if the output directory exists.

    Returns:
        str: Path to the directory containing the fine-tuned model.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED_LIST[0])

    output_dir = Config.TAPT_OUTPUT_DIR

    # Check if model already exists to avoid redundant computation
    if not force_retrain and os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"TAPT model already found at {output_dir}. Skipping training.")
        return output_dir

    print("Starting Task-Adaptive Pretraining (TAPT)...")

    # Load Tokenizer
    tokenizer = get_tokenizer()

    # Prepare Data
    # Aggregates Train and Test contexts and creates a sliding window dataset
    tapt_dataset = prepare_tapt_data(tokenizer)

    # Load Base Model for Masked Language Modeling
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_CHECKPOINT)
    model.to(Config.DEVICE)

    # Data Collator handles dynamic masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # Define Training Arguments
    # Using a temporary directory for checkpoints to keep the final output clean
    checkpoints_dir = os.path.join(Config.WORKING_DIR, "tapt_checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        learning_rate=Config.TAPT_LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        save_strategy="no",  # We only care about the final model
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=True,  # Silent execution
        seed=Config.SEED_LIST[0],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=Config.NUM_WORKERS,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tapt_dataset,
        data_collator=data_collator,
    )

    # Train
    train_result = trainer.train()

    # Print training metrics
    print("TAPT Training completed.")
    print(f"Training Loss: {train_result.training_loss}")
    if hasattr(train_result, "metrics"):
        for key, value in train_result.metrics.items():
            print(f"{key}: {value}")

    # Save the final model and tokenizer
    print(f"Saving TAPT model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
