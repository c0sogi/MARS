import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.data import prepare_train_features
from library.model import CustomXLMRoberta
from library.utils import set_seed


def get_optimizer(model):
    """
    Sets up the optimizer with Differential Learning Rates (DLR).
    - Backbone parameters get a lower learning rate to preserve pre-trained features.
    - Head parameters get a higher learning rate to learn the task quickly.
    """
    # Separate parameters
    backbone_params = list(model.backbone.parameters())

    # Combine parameters from both heads
    head_params = list(model.span_head.parameters()) + list(
        model.relevance_head.parameters()
    )

    # Define groups
    optimizer_grouped_parameters = [
        {
            "params": backbone_params,
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": head_params,
            "lr": Config.LR_HEADS,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)

    return optimizer


def compute_loss(start_logits, end_logits, relevance_logits, batch):
    """
    Computes the multi-task loss: Span Loss + Relevance Loss.
    """
    start_positions = batch["start_positions"].to(start_logits.device)
    end_positions = batch["end_positions"].to(end_logits.device)
    relevance_labels = batch["relevance_labels"].to(relevance_logits.device)

    # 1. Span Loss (Cross Entropy)
    # If the answer is not in the window (relevance=0), start/end positions are 0 (CLS).
    # The model should learn to predict 0,0 for negative samples.
    loss_fct_span = nn.CrossEntropyLoss()
    start_loss = loss_fct_span(start_logits, start_positions)
    end_loss = loss_fct_span(end_logits, end_positions)
    span_loss = (start_loss + end_loss) / 2.0

    # 2. Relevance Loss (BCE)
    loss_fct_relevance = nn.BCEWithLogitsLoss()
    relevance_loss = loss_fct_relevance(relevance_logits, relevance_labels)

    # Combined Loss
    total_loss = span_loss + (Config.RELEVANCE_LOSS_WEIGHT * relevance_loss)

    return total_loss, span_loss, relevance_loss


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    print(f"Starting Epoch {epoch + 1}/{Config.EPOCHS}")

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits, relevance_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Compute loss
        loss, span_loss, rel_loss = compute_loss(
            start_logits, end_logits, relevance_logits, batch
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Optional: Print step-level metrics if needed, but keeping it silent as requested
        # except for final metrics.

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} | Average Training Loss: {avg_loss}")

    return avg_loss


def train_seed(seed):
    """
    Orchestrates the training for a specific seed.
    Uses the full dataset (Train + Val) as per the strategy.
    """
    print(f"=== Starting Training for Seed {seed} ===")
    set_seed(seed)

    # 1. Prepare Data
    # This loads the merged train+val dataset with seed-specific negative sampling
    dataset = prepare_train_features(seed=seed, load_cached_data=True)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    device = Config.DEVICE
    model = CustomXLMRoberta(Config.MODEL_NAME)
    model.to(device)

    # 3. Setup Optimization
    optimizer = get_optimizer(model)

    # Scheduler
    num_training_steps = len(dataloader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training Loop
    # Note: No Early Stopping is implemented here because we are training on the
    # FULL dataset (Train + Val) to maximize performance for the competition.
    # There is no hold-out validation set available during this phase.
    for epoch in range(Config.EPOCHS):
        train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch)

    # 5. Save Model
    # We save the final model after all epochs
    save_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed_{seed}.bin")
    torch.save(model.state_dict(), save_path)
    print(f"Model for seed {seed} saved to {save_path}")

    # Clear memory
    del model, optimizer, scheduler, dataloader, dataset
    torch.cuda.empty_cache()


def train_all_seeds():
    """
    Iterates through all seeds defined in Config and trains a model for each.
    """
    Config.setup()

    for seed in Config.SEEDS:
        train_seed(seed)

    print("All seeds trained successfully.")
