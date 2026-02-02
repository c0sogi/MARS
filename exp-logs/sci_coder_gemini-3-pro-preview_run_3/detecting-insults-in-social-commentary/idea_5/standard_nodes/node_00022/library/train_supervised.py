import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.data_factory import create_dataloaders, get_tokenizer
from library.modeling import InsultModel


def train_seed(seed):
    """
    Trains a single InsultModel using the specified seed on the full supervised dataset.
    Loads domain-adapted weights if available.
    Saves the trained model weights to disk.

    Args:
        seed (int): The random seed to use for initialization and data shuffling.
    """
    print(f"\n{'='*40}")
    print(f"Starting Supervised Fine-Tuning for Seed: {seed}")
    print(f"{'='*40}")

    # 1. Set Seed for Reproducibility
    seed_everything(seed)

    # 2. Prepare Data
    # We use the 'supervised' stage which returns the full labeled dataset (Train + Val)
    tokenizer = get_tokenizer()
    train_dataloader = create_dataloaders(
        stage="supervised", tokenizer=tokenizer, load_cached_data=True
    )

    # 3. Initialize Model
    # Check if Domain-Adapted model exists
    if (
        os.path.exists(Config.dapt_model_output_dir)
        and len(os.listdir(Config.dapt_model_output_dir)) > 0
    ):
        print(f"Loading Domain-Adapted weights from: {Config.dapt_model_output_dir}")
        model_path = Config.dapt_model_output_dir
    else:
        print(
            f"Domain-Adapted weights not found. Loading base model: {Config.model_name}"
        )
        model_path = Config.model_name

    model = InsultModel(model_name_or_path=model_path, pretrained=True)
    model.to(Config.device)
    model.train()

    # 4. Optimizer Configuration
    # Separate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate)

    # 5. Scheduler Configuration
    num_update_steps_per_epoch = (
        len(train_dataloader) // Config.gradient_accumulation_steps
    )
    max_train_steps = Config.sft_epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_train_steps * Config.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    # 6. Loss Function
    # Binary Cross Entropy with Logits (since model outputs raw logits)
    criterion = nn.BCEWithLogitsLoss()

    # 7. Training Loop
    print(f"Training for {Config.sft_epochs} epochs...")

    for epoch in range(Config.sft_epochs):
        total_loss = 0.0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Ensure logits are squeezed to match label shape [Batch_Size]
            loss = criterion(logits.view(-1), labels)

            # Gradient Accumulation
            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % Config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * Config.gradient_accumulation_steps

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{Config.sft_epochs} | Training Loss: {avg_loss}")

    # 8. Save Model
    save_path = os.path.join(Config.working_dir, f"model_seed_{seed}.bin")
    print(f"Saving model for seed {seed} to {save_path}")
    torch.save(model.state_dict(), save_path)

    # Clear memory
    del model, optimizer, scheduler, train_dataloader
    torch.cuda.empty_cache()
    print(f"Finished training for seed {seed}.\n")
