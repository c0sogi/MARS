import time
import torch
import torch.nn as nn
from library.utils import LevenshteinMetric


class AverageMeter:
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


def train_fn(
    dataloader,
    model,
    criterion_ce,
    criterion_aux,
    optimizer,
    device,
    scheduler=None,
    aux_weight=0.2,
    epoch=0,
    pad_token_id=0,
):
    """
    Performs one epoch of training.

    Args:
        dataloader: PyTorch Dataloader.
        model: MixerTransformer model.
        criterion_ce: Cross Entropy Loss function (should have ignore_index set).
        criterion_aux: Auxiliary Loss function (e.g., MSE or SmoothL1).
        optimizer: Optimizer.
        device: torch.device.
        scheduler: Learning rate scheduler (optional).
        aux_weight: Weight for the auxiliary loss.
        epoch: Current epoch number.
        pad_token_id: ID of the padding token for masking.
    """
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_aux = AverageMeter()

    start = time.time()

    for i, batch in enumerate(dataloader):
        data_time.update(time.time() - start)

        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        atom_counts = batch["atom_counts"].to(device)

        # Prepare inputs and targets for Transformer
        # input_ids: [SOS, t1, t2, ..., EOS, PAD, ...]
        # Decoder Input: [SOS, t1, t2, ..., EOS] (remove last token/pad)
        # We slice :-1 to match the length of targets which is also shifted by 1
        decoder_input = input_ids[:, :-1]

        # Target: [t1, t2, ..., EOS, PAD] (remove first token SOS)
        targets = input_ids[:, 1:]

        # Create padding mask for decoder input (True where PAD)
        # This is used by the Transformer to ignore attention to pad tokens
        padding_mask = decoder_input == pad_token_id

        optimizer.zero_grad()

        # Forward pass
        # logits: (B, Seq_Len-1, Vocab_Size)
        # aux_preds: (B, Num_Atoms)
        logits, aux_preds = model(
            images, text_input_ids=decoder_input, padding_mask=padding_mask
        )

        # Calculate losses
        # Flatten for CrossEntropy: (B * (Seq_Len-1), Vocab_Size)
        loss_ce = criterion_ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Auxiliary Loss (Atom Counts)
        loss_aux = criterion_aux(aux_preds, atom_counts)

        # Total Loss
        loss = loss_ce + aux_weight * loss_aux

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update meters
        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_aux.update(loss_aux.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % 100 == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(dataloader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"CE {losses_ce.val:.4f} ({losses_ce.avg:.4f}) "
                f"Aux {losses_aux.val:.4f} ({losses_aux.avg:.4f})"
            )

    return losses.avg


def eval_fn(dataloader, model, tokenizer, device):
    """
    Evaluates the model using Levenshtein distance.

    Args:
        dataloader: PyTorch Dataloader (Validation set).
        model: MixerTransformer model.
        tokenizer: InchiTokenizer instance.
        device: torch.device.

    Returns:
        float: Mean Levenshtein distance.
    """
    model.eval()
    metric = LevenshteinMetric()

    start = time.time()
    print("Starting evaluation...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            targets = batch["inchi_text"]

            # Greedy decoding inference
            preds = model.generate(images, tokenizer, max_len=300, device=device)

            metric.update(preds, targets)

            if i % 50 == 0:
                print(f"Eval: [{i}/{len(dataloader)}] Time {time.time() - start:.2f}s")

    score = metric.compute()
    print(f"Validation Levenshtein Distance: {score}")
    return score
