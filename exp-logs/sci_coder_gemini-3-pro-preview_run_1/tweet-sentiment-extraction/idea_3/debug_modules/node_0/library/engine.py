import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.utils import AverageMeter, jaccard
from library.config import TweetConfig


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence loss for start and end token predictions.

    Args:
        start_logits: (Batch, Seq_Len)
        end_logits: (Batch, Seq_Len)
        start_targets: (Batch, Seq_Len) - Gaussian smoothed probabilities
        end_targets: (Batch, Seq_Len) - Gaussian smoothed probabilities
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    # Apply log_softmax to logits (KLDivLoss expects log-probabilities)
    start_log_probs = F.log_softmax(start_logits, dim=1)
    end_log_probs = F.log_softmax(end_logits, dim=1)

    start_loss = loss_fct(start_log_probs, start_targets)
    end_loss = loss_fct(end_log_probs, end_targets)

    return (start_loss + end_loss) / 2


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.amp.GradScaler("cuda")

    for data in data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_targets = data["start_tokens"].to(device)
        end_targets = data["end_tokens"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start_targets = data["start_tokens"].to(device)
            end_targets = data["end_tokens"].to(device)

            tweets = data["text"]
            offsets = data["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), input_ids.size(0))

            # --- Jaccard Calculation ---
            # Move logits to CPU for decoding
            start_logits_np = start_logits.cpu().numpy()
            end_logits_np = end_logits.cpu().numpy()

            # Get Ground Truth indices from targets (argmax of Gaussian)
            start_gt = torch.argmax(start_targets, dim=1).cpu().numpy()
            end_gt = torch.argmax(end_targets, dim=1).cpu().numpy()

            for i in range(len(tweets)):
                tweet = tweets[i]

                # 1. Decode Prediction (Joint Maximization)
                s_logits = start_logits_np[i]
                e_logits = end_logits_np[i]
                seq_len = len(s_logits)

                # Calculate sum of logits for all pairs
                sum_logits = s_logits[:, None] + e_logits[None, :]

                # Mask invalid pairs (start > end)
                # triu(k=0) keeps upper triangle including diagonal
                mask = np.triu(np.ones((seq_len, seq_len)), k=0)
                sum_logits = sum_logits * mask + (1 - mask) * -1e9

                best_idx = np.argmax(sum_logits)
                pred_start_idx = best_idx // seq_len
                pred_end_idx = best_idx % seq_len

                # 2. Extract Predicted Text
                try:
                    pred_char_start = offsets[i][pred_start_idx][0]
                    pred_char_end = offsets[i][pred_end_idx][1]
                    pred_text = tweet[pred_char_start:pred_char_end]
                except:
                    pred_text = tweet

                # 3. Extract Ground Truth Text (from target argmax)
                gt_s_idx = start_gt[i]
                gt_e_idx = end_gt[i]

                try:
                    gt_char_start = offsets[i][gt_s_idx][0]
                    gt_char_end = offsets[i][gt_e_idx][1]
                    gt_text = tweet[gt_char_start:gt_char_end]
                except:
                    gt_text = tweet

                # 4. Compute Jaccard
                score = jaccard(pred_text, gt_text)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def inference_fn(data_loader, model, device):
    """
    Runs inference on the test set using the Hybrid Strategy.
    Returns a list of predicted strings.
    """
    model.eval()
    final_predictions = []

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            tweets = data["text"]
            sentiments = data["sentiment"]
            offsets = data["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)

            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            for i in range(len(tweets)):
                tweet = tweets[i]
                sentiment = sentiments[i]

                # Hybrid Strategy: Neutral -> Full Text
                if sentiment == "neutral":
                    final_predictions.append(tweet)
                else:
                    # Positive/Negative -> Model Prediction
                    s_logits = start_logits[i]
                    e_logits = end_logits[i]
                    seq_len = len(s_logits)

                    sum_logits = s_logits[:, None] + e_logits[None, :]
                    mask = np.triu(np.ones((seq_len, seq_len)), k=0)
                    sum_logits = sum_logits * mask + (1 - mask) * -1e9

                    best_idx = np.argmax(sum_logits)
                    pred_start_idx = best_idx // seq_len
                    pred_end_idx = best_idx % seq_len

                    try:
                        char_start = offsets[i][pred_start_idx][0]
                        char_end = offsets[i][pred_end_idx][1]
                        pred_text = tweet[char_start:char_end]
                        final_predictions.append(pred_text)
                    except:
                        final_predictions.append(tweet)

    return final_predictions


def generate_submission(test_loader, model, device):
    """
    Generates predictions and saves them to the submission file.
    """
    config = TweetConfig()

    # Get predictions
    predictions = inference_fn(test_loader, model, device)

    # Load sample submission to get IDs and format
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Sanity check
    if len(predictions) != len(sample_sub):
        print(
            f"Warning: Prediction count {len(predictions)} does not match Sample count {len(sample_sub)}"
        )

    # Assign predictions
    sample_sub["selected_text"] = predictions

    # Save to CSV
    sample_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
