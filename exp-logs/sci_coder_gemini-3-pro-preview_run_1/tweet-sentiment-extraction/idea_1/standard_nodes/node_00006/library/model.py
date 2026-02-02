import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import TweetDataset, process_data
from library.utils import AverageMeter, jaccard


class TweetModel(nn.Module):
    def __init__(self, conf=Config):
        super(TweetModel, self).__init__()
        self.config = AutoConfig.from_pretrained(conf.MODEL_NAME)
        self.roberta = AutoModel.from_pretrained(conf.MODEL_NAME, config=self.config)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.config.hidden_size, 2)

        # Initialize weights for the head
        torch.nn.init.normal_(self.out.weight, std=0.02)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # DistilRoBERTa does not use token_type_ids, but we accept it for compatibility
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        out = outputs.last_hidden_state
        out = self.drop(out)
        logits = self.out(out)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    # Cite Lesson 00005: Robustness against noisy labels
    loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()

    # Use torch.amp for mixed precision (PyTorch 2.x style)
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
                    # Note: Offsets are based on "cleaned" text (normalized spaces)
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


def run_training(epochs=Config.EPOCHS, debug=Config.DEBUG):
    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)

    if debug:
        df_train = df_train.head(100)
        df_val = df_val.head(100)
        epochs = 1

    # Filter neutrals for training (Idea: Model 1)
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model Setup
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

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_jaccard = eval_fn(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f} - Val Jaccard: {val_jaccard:.5f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training complete. Best Validation Jaccard: {best_jaccard:.5f}")


def run_inference(debug=Config.DEBUG):
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
                # Neutral Heuristic
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


def main():
    run_training()
    run_inference()
