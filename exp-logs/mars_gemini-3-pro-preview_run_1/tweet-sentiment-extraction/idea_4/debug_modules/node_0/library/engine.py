import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, jaccard, AWP, seed_everything
from library.data import get_dataloaders
from library.model import TweetModel


def loss_fn(
    start_logits, end_logits, mask_logits, start_labels, end_labels, span_masks, config
):
    """
    Computes the combined loss: KLDiv for start/end + BCE for aux mask.
    """
    # KLDivLoss expects log_softmax input and probability targets
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    start_loss = loss_fct(F.log_softmax(start_logits, dim=1), start_labels)
    end_loss = loss_fct(F.log_softmax(end_logits, dim=1), end_labels)

    total_loss = start_loss + end_loss

    if config.USE_AUX_HEAD and mask_logits is not None:
        mask_loss_fct = nn.BCEWithLogitsLoss()
        # mask_logits: (Batch, Len), span_masks: (Batch, Len)
        mask_loss = mask_loss_fct(mask_logits, span_masks)
        total_loss += config.AUX_LOSS_WEIGHT * mask_loss

    return total_loss


def get_optimizer_params(model, config):
    """
    Applies Layer-wise Learning Rate Decay (LLRD).
    """
    # DeBERTa-v3 structure: model.backbone.embeddings, model.backbone.encoder.layer.{0..11}
    backbone = model.backbone
    n_layers = backbone.config.num_hidden_layers

    optimizer_parameters = []

    # 1. Head Parameters (Base LR)
    head_params = list(model.pooling.parameters()) + list(model.conv_head.parameters())
    if config.USE_AUX_HEAD:
        head_params += list(model.aux_head.parameters())

    optimizer_parameters.append(
        {
            "params": head_params,
            "lr": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
        }
    )

    # 2. Backbone Layers (Decaying LR)
    # Layer 11 (top) -> Layer 0 (bottom)
    for layer_i in range(n_layers - 1, -1, -1):
        layer_params = backbone.encoder.layer[layer_i].parameters()
        decay_power = (n_layers - 1 - layer_i) + 1  # Top layer gets decay^1
        lr = config.LEARNING_RATE * (config.LLRD_DECAY**decay_power)

        optimizer_parameters.append(
            {"params": layer_params, "lr": lr, "weight_decay": config.WEIGHT_DECAY}
        )

    # 3. Embeddings (Lowest LR)
    embed_params = backbone.embeddings.parameters()
    lr_embed = config.LEARNING_RATE * (config.LLRD_DECAY ** (n_layers + 1))
    optimizer_parameters.append(
        {"params": embed_params, "lr": lr_embed, "weight_decay": config.WEIGHT_DECAY}
    )

    # Catch any other parameters in backbone (e.g. LayerNorms outside encoder block)
    # This is a simplification; usually they are covered above or negligible.

    return optimizer_parameters


def train_fn(dataloader, model, optimizer, scheduler, awp, epoch, config, device):
    model.train()
    losses = AverageMeter()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_labels = batch["start_labels"].to(device)
        end_labels = batch["end_labels"].to(device)
        span_masks = batch["span_masks"].to(device)

        optimizer.zero_grad()

        # Forward Pass
        start_logits, end_logits, mask_logits = model(input_ids, attention_mask)

        # Calculate Loss
        loss = loss_fn(
            start_logits,
            end_logits,
            mask_logits,
            start_labels,
            end_labels,
            span_masks,
            config,
        )

        # Backward Pass
        loss.backward()

        # Adversarial Weight Perturbation (AWP)
        if config.USE_AWP and epoch >= config.AWP_START_EPOCH:
            awp.attack()
            # Forward pass with perturbed weights
            start_adv, end_adv, mask_adv = model(input_ids, attention_mask)
            loss_adv = loss_fn(
                start_adv,
                end_adv,
                mask_adv,
                start_labels,
                end_labels,
                span_masks,
                config,
            )
            # Backward pass for adversarial loss
            loss_adv.backward()
            awp.restore()

        # Optimization Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def decode_prediction(start_logits, end_logits, text, offsets, sentiment):
    """
    Decodes the span using Joint Span Decoding logic.
    """
    if sentiment == "neutral":
        return text

    start_probs = torch.softmax(start_logits, dim=0).cpu().numpy()
    end_probs = torch.softmax(end_logits, dim=0).cpu().numpy()

    # Joint decoding: maximize P(start) + P(end) s.t. start <= end
    # Create score matrix
    score_mat = np.expand_dims(start_probs, 1) + np.expand_dims(end_probs, 0)

    # Mask invalid positions (end < start)
    # np.triu returns upper triangle, we want to keep it.
    # However, we simply set lower triangle to -inf
    rows, cols = score_mat.shape
    for i in range(rows):
        for j in range(i):
            score_mat[i, j] = -1e9

    # Find max
    start_idx, end_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)

    # Convert token indices to character indices
    if start_idx >= len(offsets) or end_idx >= len(offsets):
        return text  # Fallback

    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]

    return text[char_start:char_end]


def eval_fn(dataloader, model, config, device):
    model.eval()
    jaccard_scores = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets = batch["offsets"].numpy()
            raw_texts = batch["raw_text"]
            sentiments = batch["sentiment"]
            selected_texts = batch["selected_text"]

            start_logits, end_logits, _ = model(input_ids, attention_mask)

            for i in range(len(input_ids)):
                pred_text = decode_prediction(
                    start_logits[i],
                    end_logits[i],
                    raw_texts[i],
                    offsets[i],
                    sentiments[i],
                )

                score = jaccard(selected_texts[i], pred_text)
                jaccard_scores.update(score)

    return jaccard_scores.avg


def predict_fn(dataloader, model, config, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets = batch["offsets"].numpy()
            raw_texts = batch["raw_text"]
            sentiments = batch["sentiment"]
            # Test set has no selected_text, but we need raw_text and ID (not in batch dict usually, need to track)
            # Wait, TweetDataset doesn't return ID. We assume order is preserved.
            # We will rely on the order matching the metadata file.

            start_logits, end_logits, _ = model(input_ids, attention_mask)

            for i in range(len(input_ids)):
                pred_text = decode_prediction(
                    start_logits[i],
                    end_logits[i],
                    raw_texts[i],
                    offsets[i],
                    sentiments[i],
                )
                predictions.append(pred_text)

    return predictions


def run():
    config = Config()
    seed_everything(config.SEED)

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print("Initializing Model...")
    model = TweetModel(config)
    model.to(config.DEVICE)

    # Optimizer & Scheduler
    optimizer_params = get_optimizer_params(model, config)
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=config.LEARNING_RATE, eps=config.eps, betas=config.betas
    )

    num_train_steps = int(len(train_loader) * config.EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        num_training_steps=num_train_steps,
    )

    # AWP
    awp = AWP(model, optimizer, adv_lr=config.AWP_LR, adv_eps=config.AWP_EPS)

    best_jaccard = 0.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.bin")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(config.EPOCHS):
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, awp, epoch, config, config.DEVICE
        )
        val_jaccard = eval_fn(val_loader, model, config, config.DEVICE)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best Score! Model Saved.")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation Jaccard: {best_jaccard}")

    # --- Inference on Test Set ---
    print("Generating Submission...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

    # Get predictions
    test_preds = predict_fn(test_loader, model, config, config.DEVICE)

    # Load test metadata to get IDs
    test_df = pd.read_csv(config.TEST_META_PATH)
    if config.DEBUG_SAMPLE_SIZE:
        test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

    # Create submission dataframe
    submission = pd.DataFrame(
        {"textID": test_df["textID"], "selected_text": test_preds}
    )

    # Ensure quoted format by simply saving as CSV; pandas handles quotes for strings containing delimiters.
    # The requirement "selected text needs to be quoted" usually implies standard CSV quoting for strings.
    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
