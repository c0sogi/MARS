import os
import math
import torch
from torch.utils.data import random_split
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    logging as hf_logging,
)
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_mlm_dataset

# Set transformers verbosity to error to avoid clutter
hf_logging.set_verbosity_error()


def train_mlm(model_name, load_cached_data=True):
    """
    Performs Masked Language Modeling (MLM) pre-training on a specific backbone.

    Args:
        model_name (str): The HuggingFace model identifier.
        load_cached_data (bool): If True, checks for existing checkpoints.

    Returns:
        str: Path to the fine-tuned model directory.
    """
    # Create a safe directory name from the model name
    safe_name = model_name.replace("/", "-")
    output_dir = os.path.join(Config.CHECKPOINT_DIR, f"mlm_{safe_name}")

    # Check if model is already trained and cached
    # We check for config.json and model weights
    has_config = os.path.exists(os.path.join(output_dir, "config.json"))
    has_model_bin = os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
    has_model_safe = os.path.exists(os.path.join(output_dir, "model.safetensors"))

    if load_cached_data and has_config and (has_model_bin or has_model_safe):
        print(f"Loading cached MLM model for {model_name} from {output_dir}")
        return output_dir

    print(f"Starting MLM training for {model_name}...")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        raise

    # Load Dataset (Train + Test combined)
    # get_mlm_dataset returns a PyTorch Dataset containing tokenized inputs
    full_dataset = get_mlm_dataset(tokenizer, load_cached_data=load_cached_data)

    # Split into Train (90%) and Validation (10%) for Early Stopping
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # Ensure reproducibility in split
    generator = torch.Generator().manual_seed(Config.SEED)
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"Dataset Split - Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # Data Collator for MLM
    # Handles dynamic masking of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_MASK_PROB
    )

    # Load Model
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.MLM_EPOCHS,
        per_device_train_batch_size=Config.MLM_BATCH_SIZE,
        per_device_eval_batch_size=Config.MLM_BATCH_SIZE,
        learning_rate=Config.MLM_LR,
        weight_decay=Config.MLM_WEIGHT_DECAY,
        # Evaluation and Saving
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        # Optimization
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Hardware/Env
        seed=Config.SEED,
        data_seed=Config.SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=Config.NUM_WORKERS,
        # UI
        report_to="none",
        disable_tqdm=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.PATIENCE)],
    )

    # Train
    trainer.train()

    # Evaluate final model
    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    print(f"MLM Training finished for {model_name}.")
    print(f"Final Validation Loss: {eval_metrics['eval_loss']}")
    print(f"Final Perplexity: {perplexity}")

    # Save the best model explicitly to the output directory
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def run_dapt_pipeline(load_cached_data=True):
    """
    Orchestrates the Domain-Adaptive Pre-training for all backbones defined in Config.

    Args:
        load_cached_data (bool): Whether to use cached models.

    Returns:
        dict: Mapping of original model names to their fine-tuned paths.
    """
    seed_everything()

    mlm_model_paths = {}

    for model_name in Config.MODEL_BACKBONES:
        print(f"\n--- Processing Backbone: {model_name} ---")
        try:
            path = train_mlm(model_name, load_cached_data=load_cached_data)
            mlm_model_paths[model_name] = path
        except Exception as e:
            print(f"Failed to pre-train {model_name}: {e}")
            # Fallback to original model if DAPT fails
            mlm_model_paths[model_name] = model_name

    return mlm_model_paths
