import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
import transformers
from torch.optim import AdamW

from library.config import Config
from library.utils import AverageMeter, format_time, print_metric
from library.metrics import compute_final_metric
from library.data_loader import get_data_loaders

# Suppress transformer warnings
transformers.logging.set_verbosity_error()


class ToxicityClassifier(nn.Module):
    """
    Toxicity Classifier using a DistilBERT backbone.
    Outputs a single logit for binary classification.
    """

    def __init__(self, model_name=Config.MODEL_NAME):
        super(ToxicityClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # DistilBERT returns a tuple where the first element is hidden states
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        # Use the [CLS] token embedding (first token) for classification
        cls_token = last_hidden_state[:, 0, :]

        x = self.dropout(cls_token)
        logits = self.classifier(x)
        return logits


def loss_fn(outputs, targets, weights=None):
    """
    Computes Binary Cross Entropy loss.
    If weights are provided (for bias mitigation), applies them to individual samples.
    """
    # Flatten outputs and targets
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss = criterion(outputs, targets)

    if weights is not None:
        weights = weights.view(-1)
        loss = loss * weights

    return loss.mean()


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Runs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        # Load weights if they exist (Train/Val sets)
        weights = batch["weight"].to(device) if "weight" in batch else None

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets, weights)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def validate(model, dataloader, device, df_val):
    """
    Runs validation and computes the custom competition metric.
    """
    model.eval()
    losses = AverageMeter()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            weights = batch["weight"].to(device) if "weight" in batch else None

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets, weights)

            losses.update(loss.item(), input_ids.size(0))

            # Apply sigmoid to get probabilities
            batch_preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend(batch_preds)

    # Assign predictions to the validation dataframe to calculate metrics
    # Note: DataLoader must be sequential (shuffle=False) for this to align
    df_val["prediction"] = preds

    # Compute the complex competition metric
    score = compute_final_metric(df_val, "prediction", Config.TARGET_COL, verbose=True)

    return losses.avg, score


def inference(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            batch_preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend(batch_preds)

    return preds


def run_training():
    """
    Main execution function:
    1. Sets up environment.
    2. Loads data.
    3. Trains model with freezing strategy.
    4. Evaluates and saves best model.
    5. Generates submission.
    """
    # Setup directories
    Config.setup()

    # Load DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # Load Validation DataFrame for Metric Calculation (need identity columns)
    val_df = pd.read_csv(Config.VAL_PATH)
    if Config.DEBUG:
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Initialize Model
    device = Config.DEVICE
    model = ToxicityClassifier(Config.MODEL_NAME)
    model.to(device)

    # Optimizer Groups (Weight Decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=Config.LEARNING_RATE)

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # -----------------------------------------------------------
        # Freezing Strategy:
        # Epoch 0: Freeze transformer backbone, train only head.
        # Epoch 1+: Unfreeze backbone, train all.
        # -----------------------------------------------------------
        if epoch == 0:
            print("Epoch 1: Freezing transformer backbone.")
            for param in model.base_model.parameters():
                param.requires_grad = False
        else:
            print(f"Epoch {epoch+1}: Unfreezing transformer backbone.")
            for param in model.base_model.parameters():
                param.requires_grad = True

        # Timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, device, val_df.copy())

        end_time.record()
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time) / 1000

        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {format_time(elapsed)}")
        print_metric("Train Loss", train_loss)
        print_metric("Val Loss", val_loss)

        # Checkpoint & Early Stopping
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.5f} -> {val_score:.5f}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Score did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # -----------------------------------------------------------
    # Inference & Submission
    # -----------------------------------------------------------
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    print("Generating predictions on Test set...")
    test_preds = inference(model, test_loader, device)

    # Prepare Submission DataFrame
    # If DEBUG is on, test_loader is a subset, so we must subset the sample_submission or test_df
    if Config.DEBUG:
        test_df = pd.read_csv(Config.TEST_PATH)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        submission = pd.DataFrame({"id": test_df["id"], "prediction": test_preds})
    else:
        submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        submission["prediction"] = test_preds

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
