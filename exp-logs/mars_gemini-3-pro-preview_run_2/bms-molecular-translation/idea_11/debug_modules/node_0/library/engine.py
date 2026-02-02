import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, compute_levenshtein
from library.tokenizer import InChITokenizer


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # CrossEntropyLoss ignoring the padding token
    # We assume the tokenizer's PAD_IDX is 0 based on the provided Tokenizer class
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for i, (images, labels, lengths) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with Teacher Forcing
        # Input to decoder: <SOS> ... <Last Token> (exclude <EOS> usually, or just shifted)
        # Target for loss:  <First Token> ... <EOS> (exclude <SOS>)

        # labels shape: [Batch, Seq_Len]
        # We feed labels[:, :-1] as input (0 to N-1)
        # We predict labels[:, 1:] as target (1 to N)

        optimizer.zero_grad()

        # logits shape: [Batch, Seq_Len-1, Vocab_Size]
        logits = model(images, labels[:, :-1])

        # Flatten for loss calculation
        # Targets: labels[:, 1:]
        targets = labels[:, 1:]

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg:.6f}")
    return loss_meter.avg


def validate(model, dataloader, tokenizer, device):
    """
    Evaluates the model on the validation set.
    Computes Validation Loss (Teacher Forcing) and Levenshtein Distance (Greedy Decoding).

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        tokenizer: Tokenizer instance for decoding.
        device: Device to evaluate on.

    Returns:
        dict: Dictionary containing 'val_loss' and 'val_score' (Levenshtein distance).
    """
    model.eval()
    loss_meter = AverageMeter()
    levenshtein_meter = AverageMeter()

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    # Max length for generation
    max_len = Config.MAX_TEXT_LEN

    with torch.no_grad():
        for i, (images, labels, lengths) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ---------------------------------------------------------
            # 1. Validation Loss (Teacher Forcing)
            # ---------------------------------------------------------
            logits = model(images, labels[:, :-1])
            targets = labels[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss_meter.update(loss.item(), images.size(0))

            # ---------------------------------------------------------
            # 2. Levenshtein Distance (Greedy Decoding)
            # ---------------------------------------------------------
            # We perform inference manually to avoid re-encoding images at every step

            # Encode images once -> Memory: [H*W, Batch, Dim]
            memory = model(images, tgt_seqs=None)

            batch_size = images.size(0)

            # Initialize generated sequence with SOS token
            # shape: [Batch, 1]
            ys = torch.fill(
                torch.zeros(batch_size, 1, dtype=torch.long), tokenizer.SOS_IDX
            ).to(device)

            # Storage for finished sequences
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

            for _ in range(max_len):
                # Embed current sequence
                # ys: [Batch, Curr_Len]
                tgt_emb = model.embedding(ys)  # [Batch, Curr_Len, Dim]
                tgt_emb = tgt_emb.permute(1, 0, 2)  # [Curr_Len, Batch, Dim]
                tgt_emb = model.pos_encoder_1d(tgt_emb)

                # Generate mask
                L = tgt_emb.size(0)
                tgt_mask = model.generate_square_subsequent_mask(L).to(device)

                # Decode
                # output: [Curr_Len, Batch, Dim]
                output = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

                # Project to vocab
                # Take the last token output: [Batch, Dim]
                last_output = output[-1, :, :]
                prob = model.fc_out(last_output)  # [Batch, Vocab]

                # Greedy choice
                _, next_word = torch.max(prob, dim=1)

                # Append to sequence
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

                # Check for EOS
                is_eos = next_word == tokenizer.EOS_IDX
                finished = finished | is_eos

                if finished.all():
                    break

            # Decode sequences to text
            preds = []
            ground_truths = []

            # Convert tensors to lists for tokenizer
            ys_list = ys.detach().cpu().numpy()
            labels_list = labels.detach().cpu().numpy()

            for j in range(batch_size):
                # Prediction
                pred_str = tokenizer.sequence_to_text(ys_list[j])
                preds.append(pred_str)

                # Ground Truth
                gt_str = tokenizer.sequence_to_text(labels_list[j])
                ground_truths.append(gt_str)

            # Compute Metric
            score = compute_levenshtein(preds, ground_truths)
            levenshtein_meter.update(score, batch_size)

    print(
        f"Val Loss: {loss_meter.avg:.6f} | Val Levenshtein: {levenshtein_meter.avg:.6f}"
    )

    return {"val_loss": loss_meter.avg, "val_score": levenshtein_meter.avg}
