import torch
import numpy as np
from tqdm.auto import tqdm
from library.config import Config
from library.utils import get_score
from library.optimization import AWP


def train_fn(train_loader, model, optimizer, scheduler, epoch, device):
    """
    Executes one training epoch with Mixed Precision, Gradient Accumulation, and AWP.
    """
    model.train()

    # Initialize Scaler for Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    # Initialize AWP if conditions are met
    awp = None
    if Config.use_awp and epoch >= Config.awp_start_epoch:
        awp = AWP(model, optimizer, adv_eps=Config.awp_eps, adv_lr=Config.awp_lr)

    losses = []

    # Disable progress bar output as per instructions
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # --- Standard Forward Pass ---
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]
            # Scale loss for gradient accumulation
            loss = loss / Config.gradient_accumulation_steps

        # Accumulate loss for reporting
        losses.append(loss.item() * Config.gradient_accumulation_steps)

        # --- Standard Backward Pass ---
        scaler.scale(loss).backward()

        # --- Gradient Accumulation & Optimization ---
        if (step + 1) % Config.gradient_accumulation_steps == 0:

            # --- Adversarial Weight Perturbation (AWP) ---
            if awp is not None:
                # Unscale gradients so AWP can measure true gradient norm
                scaler.unscale_(optimizer)

                # Perturb weights based on accumulated gradients
                awp.attack_step()

                # Adversarial Forward Pass (on current micro-batch)
                with torch.amp.autocast("cuda"):
                    adv_outputs = model(input_ids, attention_mask, labels=labels)
                    adv_loss = adv_outputs["loss"]
                    # We treat adversarial loss as a regularizer added to the update
                    adv_loss = adv_loss / Config.gradient_accumulation_steps

                # Adversarial Backward Pass
                scaler.scale(adv_loss).backward()

                # Restore original weights
                awp.restore()

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss}")

    return avg_loss


def eval_fn(val_loader, model, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and predictions.
    """
    model.eval()
    preds = []
    labels_list = []
    losses = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

        losses.append(loss.item())

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).float().cpu().numpy()
        preds.append(probs)
        labels_list.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    labels_list = np.concatenate(labels_list)
    avg_loss = np.mean(losses)

    # Calculate competition metric (Log Loss)
    score = get_score(labels_list, preds)

    print(f"Validation | Loss: {avg_loss} | Log Loss Score: {score}")

    return avg_loss, preds


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask, labels=None)
                logits = outputs["logits"]

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).float().cpu().numpy()
        preds.append(probs)

    preds = np.concatenate(preds)
    return preds
