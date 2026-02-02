import os
import shutil
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
from library.config import Config
from library.utils import set_seed


def compute_metrics(p):
    """
    Computes accuracy and F1 score for token classification.
    Ignores special tokens (label -100) to ensure metrics reflect
    performance on actual text.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Filter out ignored index (special tokens like PAD, CLS, SEP)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten lists for global metric computation
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    return {
        "accuracy": accuracy_score(flat_labels, flat_preds),
        "macro_f1": f1_score(flat_labels, flat_preds, average="macro"),
    }


def train_fold(train_dataset, val_dataset, seed):
    """
    Trains a single QA model instance with the specified seed.

    This function handles:
    1. Model initialization (from TAPT or Base).
    2. Trainer setup with specified hyperparameters.
    3. Training with Early Stopping.
    4. Evaluation and metric logging.
    5. Saving the best model state dict.

    Args:
        train_dataset: PyTorch Dataset for training.
        val_dataset: PyTorch Dataset for validation.
        seed (int): Random seed for initialization and shuffling.

    Returns:
        None. Saves the model artifact to Config.QA_MODELS_DIR.
    """
    # 1. Set Reproducibility Seed
    set_seed(seed)

    # 2. Determine Model Source (TAPT vs Base)
    # We prioritize the TAPT model if it was successfully generated.
    if os.path.exists(Config.TAPT_MODEL_DIR) and os.path.exists(
        os.path.join(Config.TAPT_MODEL_DIR, "config.json")
    ):
        model_name_or_path = Config.TAPT_MODEL_DIR
        print(
            f"Seed {seed}: Initializing model from TAPT weights at {model_name_or_path}"
        )
    else:
        model_name_or_path = Config.MODEL_CHECKPOINT
        print(
            f"Seed {seed}: Initializing model from base checkpoint {model_name_or_path}"
        )

    # 3. Initialize Tokenizer and Model
    # The tokenizer is required by the DataCollator for padding logic
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=3,  # Labels: 0=O, 1=B, 2=I
        ignore_mismatched_sizes=False,
    )

    # 4. Setup Training Arguments
    # We use a specific subdirectory for this seed's checkpoints
    run_output_dir = os.path.join(Config.WORKING_DIR, f"run_seed_{seed}")

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable external loggers
        disable_tqdm=True,  # Keep output clean
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    print(f"Seed {seed}: Starting training...")
    train_result = trainer.train()

    # 7. Log Metrics
    print(f"Seed {seed}: Training completed.")
    print(f"Seed {seed}: Final Training Metrics: {train_result.metrics}")

    eval_metrics = trainer.evaluate()
    print(f"Seed {seed}: Final Validation Metrics:")
    for key, value in eval_metrics.items():
        # Print full precision without formatting
        print(f"{key}: {value}")

    # 8. Save Final Model State
    # Ensure the output directory exists
    os.makedirs(Config.QA_MODELS_DIR, exist_ok=True)
    save_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")

    print(f"Seed {seed}: Saving model state dict to {save_path}")
    torch.save(model.state_dict(), save_path)

    # 9. Cleanup
    # Remove the bulky checkpoint directory to save disk space
    if os.path.exists(run_output_dir):
        print(f"Seed {seed}: Cleaning up checkpoints at {run_output_dir}")
        shutil.rmtree(run_output_dir)
