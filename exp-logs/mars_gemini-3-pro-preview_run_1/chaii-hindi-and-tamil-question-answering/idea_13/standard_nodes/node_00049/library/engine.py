import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import FGM


def get_optimizer(model):
    """
    Creates the optimizer with Differential Learning Rates and Global Weight Decay.
    Applies weight decay to ALL parameters (including bias and LayerNorm).
    """
    optimizer_parameters = [
        {
            "params": model.backbone.parameters(),
            "lr": Config.lr_backbone,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": list(model.qa_outputs.parameters())
            + list(model.relevance_classifier.parameters()),
            "lr": Config.lr_head,
            "weight_decay": Config.weight_decay,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_parameters)
    return optimizer


def get_scheduler(optimizer, num_train_steps):
    """
    Creates a linear learning rate scheduler with warmup.
    """
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )


def train_fn(dataloader, model, optimizer, device, scheduler):
    """
    Trains the model for one epoch using Adversarial Training (FGM) with Loss Normalization.
    """
    model.train()
    final_loss = 0

    # Initialize FGM if enabled
    fgm = None
    if Config.use_fgm:
        fgm = FGM(model)

    for batch in dataloader:
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        relevance = batch["relevance"].to(device)

        # --- 1. Clean Forward Pass ---
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            relevance=relevance,
        )

        loss = outputs["loss"]

        # --- 2. Backward Pass & Adversarial Training ---
        if Config.use_fgm:
            # Loss Normalization: Scale clean loss by 0.5
            (loss / 2.0).backward()

            # Attack
            fgm.attack(epsilon=Config.fgm_epsilon)

            # Adversarial Forward Pass
            outputs_adv = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                relevance=relevance,
            )
            loss_adv = outputs_adv["loss"]

            # Backward Adversarial Loss (Scaled by 0.5)
            (loss_adv / 2.0).backward()

            # Restore embeddings
            fgm.restore()
        else:
            # Standard backward if FGM is disabled
            loss.backward()

        # --- 3. Optimization Step ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        final_loss += loss.item()

    return final_loss / len(dataloader)


def eval_fn(dataloader, model, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    final_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            relevance = batch["relevance"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                relevance=relevance,
            )

            loss = outputs["loss"]
            final_loss += loss.item()

    return final_loss / len(dataloader)


def predict_fn(dataloader, model, device):
    """
    Runs inference on the provided dataloader and returns raw logits.
    """
    model.eval()

    all_start_logits = []
    all_end_logits = []
    all_relevance_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_start_logits.append(outputs["start_logits"].cpu().numpy())
            all_end_logits.append(outputs["end_logits"].cpu().numpy())
            all_relevance_logits.append(outputs["relevance_logits"].cpu().numpy())

    return (
        np.concatenate(all_start_logits),
        np.concatenate(all_end_logits),
        np.concatenate(all_relevance_logits),
    )


def train_loop(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
):
    """
    Executes the full training loop with Early Stopping logic.
    If val_dataloader is None (full train mode), saves the model at the end of every epoch.
    """
    best_loss = float("inf")

    for epoch in range(epochs):
        # Train
        train_loss = train_fn(train_dataloader, model, optimizer, device, scheduler)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}")

        # Validate (if validation set exists)
        if val_dataloader is not None:
            val_loss = eval_fn(val_dataloader, model, device)
            print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss}")

            # Early Stopping / Model Checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved Best Model at Epoch {epoch + 1}")
        else:
            # If no validation set, save the model at the end of the epoch
            # (In full training mode, we typically want the model trained for the full duration)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Model at Epoch {epoch + 1}")
