import torch
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import AverageMeter, jaccard, loss_fn


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        start_positions = d["start_positions"].to(device)
        end_positions = d["end_positions"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Jaccard score.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            start_positions = d["start_positions"].to(device)
            end_positions = d["end_positions"].to(device)

            texts = d["text"]
            selected_texts = d["selected_text"]
            sentiments = d["sentiment"]
            offsets = d["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)

            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
            losses.update(loss.item(), input_ids.size(0))

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(texts)):
                if sentiments[i] == "neutral":
                    pred_text = texts[i]
                else:
                    start_p = start_probs[i]
                    end_p = end_probs[i]

                    # Create score matrix: shape (seq_len, seq_len)
                    # We maximize P(start) + P(end)
                    score_mat = start_p[:, None] + end_p[None, :]

                    # Mask invalid spans (start > end) by setting lower triangle to a very low value
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    score_mat = np.where(upper_tri_mask == 1, score_mat, -100.0)

                    # Find index of maximum score
                    max_idx = np.argmax(score_mat)
                    best_start, best_end = np.unravel_index(max_idx, score_mat.shape)

                    # Extract text using offsets
                    start_char = offsets[i][best_start][0]
                    end_char = offsets[i][best_end][1]

                    # Handle edge cases (e.g. special tokens)
                    if best_start == best_end and start_char == 0 and end_char == 0:
                        pred_text = texts[i]
                    else:
                        pred_text = texts[i][start_char:end_char]

                score = jaccard(pred_text, selected_texts[i])
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def infer_fn(data_loader, model, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()

    # Load test metadata to get textIDs
    test_df = pd.read_csv(Config.TEST_META_PATH)
    if Config.DEBUG:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    ids = test_df["textID"].values
    prediction_strings = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            texts = d["text"]
            sentiments = d["sentiment"]
            offsets = d["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(texts)):
                if sentiments[i] == "neutral":
                    pred_text = texts[i]
                else:
                    start_p = start_probs[i]
                    end_p = end_probs[i]

                    score_mat = start_p[:, None] + end_p[None, :]
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    score_mat = np.where(upper_tri_mask == 1, score_mat, -100.0)

                    max_idx = np.argmax(score_mat)
                    best_start, best_end = np.unravel_index(max_idx, score_mat.shape)

                    start_char = offsets[i][best_start][0]
                    end_char = offsets[i][best_end][1]

                    if best_start == best_end and start_char == 0 and end_char == 0:
                        pred_text = texts[i]
                    else:
                        pred_text = texts[i][start_char:end_char]

                    if len(pred_text.strip()) == 0:
                        pred_text = texts[i]

                prediction_strings.append(pred_text)

    # Create submission dataframe
    submission = pd.DataFrame({"textID": ids, "selected_text": prediction_strings})

    # Save to CSV
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
