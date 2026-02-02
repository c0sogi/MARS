import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, get_device, AverageMeter
from library.data_processing import get_mlm_dataset


def run_tapt(
    model_name: str,
    output_path: str,
    debug: bool = Config.DEBUG,
    epochs: int = Config.TAPT_PARAMS["epochs"],
    batch_size: int = Config.TAPT_PARAMS["batch_size"],
    learning_rate: float = Config.TAPT_PARAMS["learning_rate"],
    weight_decay: float = Config.TAPT_PARAMS["weight_decay"],
    mlm_probability: float = Config.TAPT_PARAMS["mlm_probability"],
):
    """
    Executes Task-Adaptive Pretraining (TAPT) using Masked Language Modeling (MLM).

    This function trains a transformer model on the domain corpus (combined train and test text)
    using an unsupervised MLM objective. The resulting weights are saved to disk to be used
    as the initialization for the supervised classification task.

    Args:
        model_name (str): The name of the HuggingFace model to pretrain (e.g., 'roberta-base').
        output_path (str): Directory path where the adapted model weights will be saved.
        debug (bool): If True, runs with a smaller subset of data for debugging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.
        mlm_probability (float): Probability of masking a token for MLM.
    """
    print(f"--- Starting TAPT for {model_name} ---")

    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = get_device()
    os.makedirs(output_path, exist_ok=True)

    # 2. Load Tokenizer
    # We need the tokenizer to prepare the dataset
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Prepare Dataset
    # get_mlm_dataset handles loading raw text and creating the dataset object
    print("Preparing MLM dataset...")
    dataset = get_mlm_dataset(tokenizer, debug=debug)

    # 4. Data Collator
    # Handles dynamic masking of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    # 5. DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # 6. Load Model
    # We use AutoModelForMaskedLM for the pretraining objective
    print(f"Loading model {model_name} for Masked LM...")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    # 7. Optimizer and Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scaler = GradScaler()

    num_training_steps = len(dataloader) * epochs
    # Using a linear schedule is standard for BERT-like pretraining
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * num_training_steps),  # ~6% warmup
        num_training_steps=num_training_steps,
    )

    # 8. Training Loop
    print(f"Starting training for {epochs} epochs...")
    model.train()

    global_step = 0

    for epoch in range(epochs):
        loss_meter = AverageMeter()

        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            with autocast(enabled=True):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), Config.TRAIN_PARAMS["max_grad_norm"]
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Logging
            loss_meter.update(loss.item(), input_ids.size(0))
            global_step += 1

            if step % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{step}/{len(dataloader)}] "
                    f"Loss: {loss_meter.val:.8f} (Avg: {loss_meter.avg:.8f})"
                )

        print(f"Epoch {epoch + 1} completed. Average Loss: {loss_meter.avg:.8f}")

    # 9. Save Model
    print(f"Saving TAPT model to {output_path}...")
    # save_pretrained saves config.json and pytorch_model.bin
    # This directory can be passed to AutoModel.from_pretrained() later
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("TAPT completed successfully.")
