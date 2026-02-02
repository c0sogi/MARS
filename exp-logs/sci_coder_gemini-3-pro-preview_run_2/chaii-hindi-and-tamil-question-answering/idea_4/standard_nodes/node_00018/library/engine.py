import torch
import numpy as np
from library.config import Config


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes the training loop for one epoch.

    Args:
        data_loader: PyTorch DataLoader for training data.
        model: The QA model.
        optimizer: The optimizer.
        device: The device to run on (cuda/cpu).
        scheduler: Optional learning rate scheduler.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = []

    for batch in data_loader:
        # Move batch data to device
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        start_positions = batch["start_positions"].to(device, dtype=torch.long)
        end_positions = batch["end_positions"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    print(f"Average Training Loss: {avg_loss}")
    return avg_loss


def eval_fn(data_loader, model, device):
    """
    Executes the evaluation loop to generate predictions.

    Args:
        data_loader: PyTorch DataLoader for validation/test data.
        model: The QA model.
        device: The device to run on.

    Returns:
        tuple: (all_start_logits, all_end_logits) as numpy arrays.
    """
    model.eval()
    start_logits_list = []
    end_logits_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)

            # Mixed precision inference
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Collect logits
            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

            start_logits_list.append(start_logits)
            end_logits_list.append(end_logits)

    # Concatenate results from all batches
    if len(start_logits_list) > 0:
        all_start_logits = np.concatenate(start_logits_list, axis=0)
        all_end_logits = np.concatenate(end_logits_list, axis=0)
    else:
        all_start_logits = np.array([])
        all_end_logits = np.array([])

    return all_start_logits, all_end_logits
