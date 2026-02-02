import os
import torch
import math
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from library.config import Config
from library.utils import set_seed
from library.model_factory import get_tokenizer, get_tapt_model
from library.data_manager import prepare_tapt_corpus


class TextDataset(Dataset):
    """
    Simple Dataset for loading text lines for MLM training.
    """

    def __init__(self, tokenizer, file_path, max_length):
        self.examples = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Tokenize all lines
            # We use batch_encode_plus for efficiency
            tokenized = tokenizer(
                lines,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=False,  # MLM collator creates this
                return_token_type_ids=False,
            )
            self.examples = tokenized["input_ids"]
        else:
            print(f"Warning: File {file_path} not found.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.examples[i], dtype=torch.long)}


def run_tapt():
    """
    Runs Task-Adaptive Pretraining (MLM) on the combined corpus.
    Saves the adapted model to Config.TAPT_MODEL_PATH.
    """
    set_seed(Config.SEED)

    # 1. Prepare Data
    corpus_path = os.path.join(Config.TAPT_CACHE_DIR, "corpus.txt")
    prepare_tapt_corpus(corpus_path, load_cached_data=True)

    tokenizer = get_tokenizer()
    dataset = TextDataset(tokenizer, corpus_path, Config.MAX_LENGTH)

    if len(dataset) == 0:
        print("No data found for TAPT. Skipping.")
        return

    # 2. Model
    model = get_tapt_model()

    # 3. Training Setup
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.TAPT_MLM_PROB
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(Config.WORKING_DIR, "tapt_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=Config.TAPT_EPOCHS,
        per_device_train_batch_size=Config.TAPT_BATCH_SIZE,
        learning_rate=Config.TAPT_LEARNING_RATE,
        save_steps=500,
        save_total_limit=2,
        seed=Config.SEED,
        disable_tqdm=True,  # Reduce noise
        report_to="none",
        logging_strategy="epoch",
        dataloader_num_workers=Config.NUM_WORKERS,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting Task-Adaptive Pretraining (TAPT)...")
    trainer.train()

    print(f"Saving TAPT model to {Config.TAPT_MODEL_PATH}...")
    trainer.save_model(Config.TAPT_MODEL_PATH)
    tokenizer.save_pretrained(Config.TAPT_MODEL_PATH)


def train_qa_fold(model, train_loader, val_loader, fold_idx):
    """
    Trains the QA model for a single fold using a custom PyTorch loop.
    Implements Early Stopping and saves the best model.
    """
    # Removed set_seed(Config.SEED) to allow external seed control for ensembling
    device = Config.DEVICE
    model.to(device)

    # Optimizer
    # Separate weight decay for bias/LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    # Scheduler
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = Config.EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(0.1 * max_train_steps)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    loss_fct = CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(
        Config.QA_MODEL_OUTPUT_DIR, f"model_fold_{fold_idx}.pt"
    )

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            # Flatten logits: [batch_size * seq_len, num_labels]
            # Flatten labels: [batch_size * seq_len]
            loss = loss_fct(logits.view(-1, Config.NUM_LABELS), labels.view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fct(logits.view(-1, Config.NUM_LABELS), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss}"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print(f"Fold {fold_idx} finished. Best Validation Loss: {best_val_loss}")

    # Free memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
