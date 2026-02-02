import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.awp import AWP
from library.utils import compute_pearson_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AWPBCEWrapper(nn.Module):
    """
    Wraps the PhraseModel to enforce BCEWithLogitsLoss.
    The provided AWP class relies on 'outputs.loss', which defaults to MSE
    in Hugging Face models for regression (num_labels=1).
    This wrapper computes BCE manually and attaches it to the output.
    """

    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Call model with labels=None to prevent internal MSE calculation
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
        )

        loss = None
        if labels is not None:
            # Compute BCEWithLogitsLoss
            loss = self.criterion(outputs.logits.view(-1), labels.view(-1))

        # Create a proxy object that mimics the expected output structure for AWP
        class OutputProxy:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

            # Allow indexing to support 'outputs[0]' access pattern if needed
            def __getitem__(self, idx):
                if idx == 0:
                    return self.logits
                raise IndexError

        return OutputProxy(outputs.logits, loss)


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, scaler=None):
    """
    Performs one epoch of training with AWP and Mixed Precision.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    # Wrap model to ensure BCE loss is used during AWP attack steps
    wrapped_model = AWPBCEWrapper(model, criterion)

    # Initialize AWP
    awp = None
    if Config.use_awp:
        awp = AWP(
            wrapped_model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
            scaler=scaler,
        )

    losses = AverageMeter()

    for step, batch in enumerate(train_loader):
        # Prepare inputs: filter out 'id' and move to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}
        batch_size = inputs["input_ids"].size(0)

        # Forward Pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=True):
            outputs = wrapped_model(**inputs)
            loss = outputs.loss

        losses.update(loss.item(), batch_size)

        # Backward Pass
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation
        # awp.attack_backward handles the save -> perturb -> forward -> backward -> restore cycle
        if awp is not None:
            awp.attack_backward(inputs, epoch)

        # Optimizer Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if (step + 1) % Config.print_freq == 0 or (step + 1) == len(train_loader):
            lr = scheduler.get_last_lr()[0] if scheduler else Config.learning_rate
            print(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {lr:.8f}"
            )

    return losses.avg


def eval_fn(valid_loader, model, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses = AverageMeter()
    preds = []
    labels_list = []

    for step, batch in enumerate(valid_loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}
        batch_size = inputs["input_ids"].size(0)
        labels = inputs["labels"]

        with torch.no_grad():
            # Pass labels=None to manually compute BCE
            outputs = model(inputs["input_ids"], inputs["attention_mask"], labels=None)
            logits = outputs.logits.view(-1)
            loss = criterion(logits, labels.view(-1))

        losses.update(loss.item(), batch_size)

        # Apply sigmoid to convert logits to 0-1 scores
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        labels_list.append(labels.cpu().numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(labels_list)

    score = compute_pearson_score(ground_truth, predictions)

    print(f"Validation Loss: {losses.avg:.8f}")
    print(f"Validation Pearson: {score:.8f}")

    return losses.avg, score


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    for step, batch in enumerate(test_loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}

        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"], labels=None)
            logits = outputs.logits.view(-1)

        # Apply sigmoid for final score
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    predictions = np.concatenate(preds)
    return predictions
