import os
import torch
import transformers
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from library.config import Config
from library.utils import set_seed
from library.dataset import prepare_mlm_data, get_tokenizer


def run_dapt(load_cached_data=True):
    """
    Executes Domain-Adaptive Pre-Training (DAPT) using Masked Language Modeling (MLM).

    This function adapts the generic DeBERTa-v3 backbone to the specific StackExchange
    domain by training on the concatenated text from training, validation, and test sets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed dataset from cache.
                                 If False or cache miss, processes data from scratch.
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    transformers.logging.set_verbosity_error()

    output_path = Config.DAPT_MODEL_OUTPUT_PATH

    # Check for existing model to optimize runtime
    # We check for the presence of the model weights file
    if os.path.exists(output_path):
        weights_bin = os.path.join(output_path, "pytorch_model.bin")
        weights_safe = os.path.join(output_path, "model.safetensors")
        if os.path.exists(weights_bin) or os.path.exists(weights_safe):
            print(f"DAPT model found at {output_path}. Skipping training phase.")
            return

    print(f"Initializing Domain-Adaptive Pre-Training for {Config.MODEL_DEBERTA}...")

    # 2. Prepare Data
    tokenizer = get_tokenizer(Config.MODEL_DEBERTA)

    # Load dataset (handles caching internally via library.dataset)
    dataset = prepare_mlm_data(load_cached_data=load_cached_data, tokenizer=tokenizer)
    print(f"MLM Dataset loaded. Total samples: {len(dataset)}")

    # 3. Initialize Model
    # Load backbone with a Masked Language Modeling head
    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_DEBERTA)

    # 4. Configure Training
    # Data Collator handles dynamic masking of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.DAPT_MLM_PROB
    )

    # Temporary directory for checkpoints during training
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "dapt_checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.DAPT_EPOCHS,
        per_device_train_batch_size=Config.DAPT_BATCH_SIZE,
        gradient_accumulation_steps=Config.DAPT_GRAD_ACCUM_STEPS,
        learning_rate=Config.DAPT_LR,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",  # We only save the final model
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=4,
        seed=Config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # 5. Execute Training
    print("Starting MLM training...")
    train_result = trainer.train()

    print("MLM Training completed.")
    print(f"Final Training Loss: {train_result.training_loss}")
    print(f"Total Steps: {train_result.global_step}")

    # 6. Save Artifacts
    print(f"Saving adapted model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    print("DAPT module execution finished successfully.")
