import os
import torch
from torch.optim import AdamW
from transformers import RobertaForMaskedLM, get_linear_schedule_with_warmup
from library.config import Config
from library.utils import seed_everything
from library.data_factory import create_dataloaders, get_tokenizer


def run_domain_adaptation():
    """
    Executes the Domain-Adaptive Pre-training (DAPT) stage.
    Trains RobertaForMaskedLM on the combined (Train + Val + Test) dataset
    using the Masked Language Modeling (MLM) objective.
    Saves the adapted model to Config.dapt_model_output_dir.
    """
    print("Starting Domain-Adaptive Pre-training (DAPT)...")

    # 1. Reproducibility
    seed_everything(Config.seeds[0])

    # 2. Prepare Data
    tokenizer = get_tokenizer()
    train_dataloader = create_dataloaders(
        stage="dapt", tokenizer=tokenizer, load_cached_data=True
    )

    # 3. Initialize Model
    # We use RobertaForMaskedLM to train on the MLM objective
    print(f"Loading {Config.model_name} for Masked Language Modeling...")
    model = RobertaForMaskedLM.from_pretrained(Config.model_name)
    model.to(Config.device)
    model.train()

    # 4. Optimizer and Scheduler
    # Only optimizing parameters that require gradients (though usually all are trainable in DAPT)
    optimizer = AdamW(
        model.parameters(),
        lr=Config.dapt_learning_rate,
        weight_decay=Config.weight_decay,
    )

    total_steps = len(train_dataloader) * Config.dapt_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * Config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # 5. Training Loop
    print(f"Training for {Config.dapt_epochs} epochs on device: {Config.device}")

    for epoch in range(Config.dapt_epochs):
        total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            # Forward pass
            # RobertaForMaskedLM computes loss automatically if labels are provided
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{Config.dapt_epochs} | Average MLM Loss: {avg_loss}")

    # 6. Save Adapted Model
    # We save the pretrained model so it can be loaded by AutoModel.from_pretrained later
    print(f"Saving domain-adapted model to {Config.dapt_model_output_dir}...")
    os.makedirs(Config.dapt_model_output_dir, exist_ok=True)

    model.save_pretrained(Config.dapt_model_output_dir)
    tokenizer.save_pretrained(Config.dapt_model_output_dir)

    print("DAPT complete.")
