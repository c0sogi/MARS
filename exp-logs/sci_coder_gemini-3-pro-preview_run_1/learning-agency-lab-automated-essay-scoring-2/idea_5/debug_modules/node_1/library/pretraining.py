import os
import shutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from library.config import Config
from library.utils import seed_everything
from library.data import get_mlm_data


def run_mlm(load_cached_data=True):
    """
    Executes the Domain-Adaptive Pre-training (DAPT) stage using Masked Language Modeling.
    Trains the backbone on the combined (Train + Test) corpus to adapt to the student essay domain.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        str: The directory path where the adapted model is saved.
    """
    # 1. Setup Environment
    seed_everything(Config.seed)
    print(f"Starting Domain-Adaptive Pre-training (MLM) on device: {Config.device}")

    # Define output directory for the final adapted model
    mlm_output_dir = os.path.join(Config.output_dir, "deberta_mlm")

    # 2. Initialize Tokenizer
    # We use the base model name to load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 3. Load Data
    # get_mlm_data returns an EssayDataset containing combined text from train and test sets
    train_dataset = get_mlm_data(tokenizer, load_cached_data=load_cached_data)
    print(f"MLM Dataset size: {len(train_dataset)} samples")

    # 4. Initialize Model
    # Load model for Masked Language Modeling
    # Note: For DeBERTa-v3, this initializes a MaskedLM head which may be randomly initialized
    # if the checkpoint is a discriminator. This is expected for DAPT.
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.config.use_cache = False
    model.to(Config.device)

    # 5. Data Collator
    # Handles dynamic padding and random masking of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
    )

    # 6. Training Arguments
    # Use a temporary directory for checkpoints
    checkpoints_dir = os.path.join(Config.output_dir, "mlm_checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.mlm_epochs,
        per_device_train_batch_size=Config.mlm_batch_size,
        learning_rate=Config.mlm_learning_rate,
        weight_decay=Config.weight_decay,
        fp16=Config.use_fp16,
        save_strategy="no",  # We only need the final model
        logging_strategy="epoch",
        report_to="none",  # Disable external logging
        disable_tqdm=True,  # Silent execution
        dataloader_num_workers=Config.num_workers,
        seed=Config.seed,
        gradient_checkpointing=True,  # Critical for memory management with Large models
        optim="adamw_torch",
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 8. Execute Training
    print("Initiating MLM training...")
    train_result = trainer.train()

    print("MLM Training completed.")
    print(f"Final Training Loss: {train_result.training_loss}")

    # 9. Save Adapted Model
    # Save the model and tokenizer so they can be loaded via AutoModel.from_pretrained in the next stage
    print(f"Saving adapted backbone to {mlm_output_dir}...")
    trainer.save_model(mlm_output_dir)
    tokenizer.save_pretrained(mlm_output_dir)

    # Clean up intermediate checkpoints to save space
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)

    return mlm_output_dir
