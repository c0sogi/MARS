import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.utils import AverageMeter, jaccard
from library.config import Config


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the Cross Entropy Loss with Label Smoothing for start and end indices.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return start_loss + end_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training using Mixed Precision (AMP).
    """
    model.train()
    losses = AverageMeter()

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

    for d in data_loader:
        input_ids = d["ids"].to(device)
        attention_mask = d["mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        targets_start = d["targets_start"].to(device)
        targets_end = d["targets_end"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end)

        scaler.scale(loss).backward()

        # Unscale and clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set and computes the Jaccard score.
    Loads validation metadata to compare against ground truth 'selected_text'.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    # Load validation metadata for ground truth comparison
    # We create a lookup map: textID -> selected_text
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_val["selected_text"] = df_val["selected_text"].fillna("")
    id_to_target = dict(zip(df_val["textID"], df_val["selected_text"]))

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets_start = d["targets_start"].to(device)
            targets_end = d["targets_end"].to(device)

            # Metadata for reconstruction
            orig_tweets = d["orig_tweet"]
            sentiments = d["sentiment"]
            offsets = d["offsets"]  # Keep on CPU
            text_ids = d["textID"]

            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = loss_fn(start_logits, end_logits, targets_start, targets_end)
            losses.update(loss.item(), input_ids.size(0))

            # Get predicted indices
            start_preds = torch.argmax(start_logits, dim=1).cpu().detach().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().detach().numpy()

            # Reconstruct strings and compute Jaccard
            for i in range(len(input_ids)):
                text_id = text_ids[i]
                text = orig_tweets[i]
                sentiment = sentiments[i]
                offset = offsets[i].numpy()

                pred_start = start_preds[i]
                pred_end = end_preds[i]

                # Neutral Sentiment Override
                if sentiment == "neutral":
                    predicted_text = text
                else:
                    if pred_start > pred_end:
                        pred_end = pred_start

                    # Extract substring using offsets
                    if pred_start < len(offset) and pred_end < len(offset):
                        start_char = offset[pred_start][0]
                        end_char = offset[pred_end][1]
                        predicted_text = text[start_char:end_char]
                    else:
                        predicted_text = text  # Fallback

                target_text = id_to_target.get(text_id, "")
                score = jaccard(predicted_text, target_text)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def generate_submission(test_loader, models, device):
    """
    Generates predictions for the test set using an ensemble of models.
    Averages logits from all provided models before decoding.
    Saves the result to submission.csv.
    """
    # Ensure all models are in eval mode and on device
    for m in models:
        m.eval()
        m.to(device)

    predictions = []

    with torch.no_grad():
        for d in test_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            orig_tweets = d["orig_tweet"]
            sentiments = d["sentiment"]
            offsets = d["offsets"]
            text_ids = d["textID"]

            # Ensemble: Sum logits from all models
            avg_start_logits = None
            avg_end_logits = None

            for model in models:
                start_logits, end_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                if avg_start_logits is None:
                    avg_start_logits = start_logits
                    avg_end_logits = end_logits
                else:
                    avg_start_logits += start_logits
                    avg_end_logits += end_logits

            # Average logits
            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            start_preds = torch.argmax(avg_start_logits, dim=1).cpu().detach().numpy()
            end_preds = torch.argmax(avg_end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                text = orig_tweets[i]
                sentiment = sentiments[i]
                offset = offsets[i].numpy()
                pred_start = start_preds[i]
                pred_end = end_preds[i]

                if sentiment == "neutral":
                    pred_text = text
                else:
                    if pred_start > pred_end:
                        pred_end = pred_start

                    if pred_start < len(offset) and pred_end < len(offset):
                        start_char = offset[pred_start][0]
                        end_char = offset[pred_end][1]
                        pred_text = text[start_char:end_char]
                    else:
                        pred_text = text

                predictions.append({"textID": text_ids[i], "selected_text": pred_text})

    # Save submission
    df_sub = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
