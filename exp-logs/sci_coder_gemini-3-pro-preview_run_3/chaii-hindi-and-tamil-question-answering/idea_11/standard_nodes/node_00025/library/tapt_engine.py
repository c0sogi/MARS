import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from library.config import Config
from library.utils import set_seed
from library.data import prepare_tapt_data, TAPTDataset


def run_tapt_pretraining(load_cached_data: bool = True):
    """
    Executes the Question-Context Task-Adaptive Pretraining (QC-TAPT) pipeline.

    1. Loads the base XLM-R model.
    2. Prepares the corpus (Question + Context pairs) from Train, Val, and Test sets.
    3. Fine-tunes the model using Masked Language Modeling (MLM).
    4. Saves the adapted model to disk.

    Args:
        load_cached_data (bool): Whether to load pre-processed text data from cache.
    """
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Starting TAPT Pretraining on device: {Config.DEVICE}")

    # Ensure output directories exist
    os.makedirs(Config.TAPT_MODEL_DIR, exist_ok=True)

    # 2. Load Tokenizer and Model
    # We use the base model for TAPT initialization
    print(f"Loading tokenizer and model: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_NAME)
    model.to(Config.DEVICE)

    # 3. Prepare Data
    # Get list of "Question </s> Context" strings
    texts = prepare_tapt_data(tokenizer, load_cached_data=load_cached_data)

    if Config.DEBUG:
        print("DEBUG mode: Truncating TAPT dataset to 50 examples.")
        texts = texts[:50]

    dataset = TAPTDataset(texts=texts, tokenizer=tokenizer, max_length=Config.MAX_LEN)

    # Data Collator handles the masking (MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # 4. Training Arguments
    # We use a temporary directory for checkpoints during training
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "tapt_checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        learning_rate=Config.TAPT_LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        save_strategy="no",  # We save explicitly at the end to save space
        logging_strategy="epoch",
        report_to="none",  # Disable wandb/mlflow
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        disable_tqdm=False,
        seed=Config.SEED,
        data_seed=Config.SEED,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 6. Train
    print("Starting training...")
    train_result = trainer.train()

    # Print metrics with full precision
    print("Training completed.")
    print(f"Global Step: {train_result.global_step}")
    print(f"Training Loss: {train_result.training_loss}")

    # 7. Save Model
    print(f"Saving TAPT model to {Config.TAPT_MODEL_DIR}...")
    trainer.save_model(Config.TAPT_MODEL_DIR)
    tokenizer.save_pretrained(Config.TAPT_MODEL_DIR)
    print("TAPT pipeline finished successfully.")
