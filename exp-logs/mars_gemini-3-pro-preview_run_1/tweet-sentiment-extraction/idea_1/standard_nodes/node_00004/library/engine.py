import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.utils import AverageMeter, jaccard
from library.dataset import TweetDataset, process_data
from library.model import TweetModel, loss_fn


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    # Initialize scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    for d in data_loader:
        input_ids = d["ids"].to(device)
        attention_mask = d["mask"].to(device)
        targets_start = d["targets_start"].to(device)
        targets_end = d["targets_end"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            start_logits, end_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            loss = loss_fn(start_logits, end_logits, targets_start, targets_end)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set using Jaccard score.
    """
    model.eval()
    jaccards = AverageMeter()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            offsets = d["offsets"].numpy()
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            sentiment = d["sentiment"]

            start_logits, end_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                # Heuristic: If neutral, predict full text
                if sentiment[i] == "neutral":
                    pred_text = orig_tweet[i]
                else:
                    idx_start = np.argmax(start_probs[i])
                    idx_end = np.argmax(end_probs[i])

                    if idx_start > idx_end:
                        idx_end = idx_start

                    cur_offsets = offsets[i]
                    # Ensure indices are within bounds
                    if idx_start >= len(cur_offsets):
                        idx_start = len(cur_offsets) - 1
                    if idx_end >= len(cur_offsets):
                        idx_end = len(cur_offsets) - 1

                    # Reconstruct text from offsets
                    text_clean = " " + " ".join(orig_tweet[i].split())
                    char_start = cur_offsets[idx_start][0]
                    char_end = cur_offsets[idx_end][1]

                    if char_start < len(text_clean) and char_end <= len(text_clean):
                        pred_text = text_clean[char_start:char_end]
                    else:
                        pred_text = orig_tweet[i]

                    pred_text = pred_text.strip()

                score = jaccard(pred_text, orig_selected[i])
                jaccards.update(score)

    return jaccards.avg


def run_training(epochs=Config.EPOCHS, patience=3, debug=Config.DEBUG):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)

    if debug:
        df_train = df_train.head(100)
        df_val = df_val.head(100)
        epochs = 2

    # Filter neutrals for training (Idea: Model 1 Strategy)
    df_train_filtered = df_train[df_train["sentiment"] != "neutral"].reset_index(
        drop=True
    )

    # Process Data
    train_data = process_data(
        df_train_filtered,
        tokenizer,
        Config.MAX_LEN,
        Config.WORKING_DIR,
        prefix="train_filt",
        load_cached_data=True,
        debug=debug,
    )
    val_data = process_data(
        df_val,
        tokenizer,
        Config.MAX_LEN,
        Config.WORKING_DIR,
        prefix="val",
        load_cached_data=True,
        debug=debug,
    )

    train_dataset = TweetDataset(train_data)
    val_dataset = TweetDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = TweetModel(Config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    num_train_steps = int(len(train_dataset) / Config.TRAIN_BATCH_SIZE * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_jaccard = 0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_jaccard = eval_fn(val_loader, model, device)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Jaccard: {val_jaccard}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Validation Jaccard: {best_jaccard}")


def run_inference(debug=Config.DEBUG):
    """
    Generates predictions for the test set and saves the submission file.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    df_test = pd.read_csv(Config.TEST_META_PATH)
    if debug:
        df_test = df_test.head(100)

    test_data = process_data(
        df_test,
        tokenizer,
        Config.MAX_LEN,
        Config.WORKING_DIR,
        prefix="test",
        load_cached_data=True,
        debug=debug,
    )
    test_dataset = TweetDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = TweetModel(Config)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model checkpoint not found. Using random weights.")

    model.to(device)
    model.eval()

    final_ids = []
    final_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for d in test_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            offsets = d["offsets"].numpy()
            orig_tweet = d["orig_tweet"]
            sentiment = d["sentiment"]
            text_ids = d["textID"]

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                if sentiment[i] == "neutral":
                    pred_text = orig_tweet[i]
                else:
                    idx_start = np.argmax(start_probs[i])
                    idx_end = np.argmax(end_probs[i])

                    if idx_start > idx_end:
                        idx_end = idx_start

                    cur_offsets = offsets[i]
                    if idx_start >= len(cur_offsets):
                        idx_start = len(cur_offsets) - 1
                    if idx_end >= len(cur_offsets):
                        idx_end = len(cur_offsets) - 1

                    text_clean = " " + " ".join(orig_tweet[i].split())
                    char_start = cur_offsets[idx_start][0]
                    char_end = cur_offsets[idx_end][1]

                    if char_start < len(text_clean) and char_end <= len(text_clean):
                        pred_text = text_clean[char_start:char_end]
                    else:
                        pred_text = orig_tweet[i]

                    pred_text = pred_text.strip()

                final_ids.append(text_ids[i])
                final_preds.append(pred_text)

    submission = pd.DataFrame({"textID": final_ids, "selected_text": final_preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
