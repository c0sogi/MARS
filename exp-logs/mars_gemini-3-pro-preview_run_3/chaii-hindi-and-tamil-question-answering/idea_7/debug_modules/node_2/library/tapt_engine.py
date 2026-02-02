import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from library.config import Config
from library.data import TAPTDataset
from library.utils import set_seed


def run_tapt():
    """
    Executes Task-Adaptive Pretraining (TAPT) using Masked Language Modeling (MLM).

    This function fine-tunes the base XLM-RoBERTa model on the combined text
    from the training, validation, and test sets to adapt the model's
    embeddings to the specific Hindi/Tamil vocabulary and domain context.

    Returns:
        str: The path to the directory containing the fine-tuned model and tokenizer.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print("Initializing TAPT (Masked Language Modeling)...")

    # 1. Prepare Dataset
    # TAPTDataset automatically handles loading, tokenization, sliding windows, and caching.
    # It aggregates context text from Train, Val, and Test splits.
    tapt_dataset = TAPTDataset(load_cached_data=Config.LOAD_CACHED_DATA)
    print(f"TAPT Dataset loaded. Total sequences: {len(tapt_dataset)}")

    # 2. Load Model and Tokenizer
    # We start with the base pre-trained checkpoint.
    print(f"Loading base model architecture: {Config.MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_CHECKPOINT)

    # 3. Data Collator
    # Handles dynamic masking of tokens for MLM objective.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_PROBABILITY
    )

    # 4. Configure Training Arguments
    # We use a temporary checkpoint directory for intermediate states,
    # but the final model will be saved to TAPT_OUTPUT_DIR.
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "tapt_checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        learning_rate=Config.TAPT_LEARNING_RATE,
        weight_decay=0.01,
        save_strategy="no",  # We save manually at the end
        logging_strategy="epoch",
        report_to="none",  # Disable wandb/mlflow
        disable_tqdm=True,  # clean output
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=Config.NUM_WORKERS,
        seed=Config.SEED,
        data_seed=Config.SEED,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tapt_dataset,
    )

    # 6. Run Training
    print("Starting TAPT fine-tuning...")
    train_result = trainer.train()

    print("TAPT Training Complete.")
    print(f"Global Step: {train_result.global_step}")
    print(f"Training Loss: {train_result.training_loss}")

    # 7. Save Artifacts
    print(f"Saving fine-tuned TAPT model to {Config.TAPT_OUTPUT_DIR}...")
    trainer.save_model(Config.TAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.TAPT_OUTPUT_DIR)

    # Clean up GPU memory
    del model
    del trainer
    torch.cuda.empty_cache()

    return Config.TAPT_OUTPUT_DIR
