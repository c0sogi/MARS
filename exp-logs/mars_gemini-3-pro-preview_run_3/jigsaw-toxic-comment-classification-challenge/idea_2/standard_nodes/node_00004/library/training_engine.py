import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.utils import seed_everything, calculate_roc_auc
from library.data_processing import (
    load_data_from_metadata,
    get_tfidf_features,
    make_dataloaders,
)
from library.model_definitions import NBSVM, ToxicRoBERTa

# Constants
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBMISSION_DIR = "./submission"


def train_nbsvm_pipeline(train_df, val_df, test_df, load_cached_data=True):
    """
    Trains the NBSVM model and generates predictions.
    """
    print("\n=== Starting NBSVM Pipeline ===")

    # 1. Get Features
    train_feats, val_feats, test_feats = get_tfidf_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 2. Prepare Targets
    y_train = train_df[LABEL_COLS].values
    y_val = val_df[LABEL_COLS].values

    # 3. Train Model
    print("Training NBSVM model...")
    model = NBSVM(C=1.0, dual=True, n_jobs=-1)
    model.fit(train_feats, y_train)

    # 4. Validate
    print("Validating NBSVM...")
    val_probs = model.predict_proba(val_feats)
    val_auc = calculate_roc_auc(y_val, val_probs)
    print(f"NBSVM Validation ROC AUC: {val_auc}")

    # 5. Inference
    print("Generating NBSVM test predictions...")
    test_probs = model.predict_proba(test_feats)

    return test_probs, val_auc


def train_roberta_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Runs one training epoch for RoBERTa.
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["targets"].to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(ids, mask)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_roberta(model, dataloader, device, criterion):
    """
    Evaluates RoBERTa on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    total_loss = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    auc_score = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc_score


def infer_roberta(model, dataloader, device):
    """
    Generates predictions for the test set using RoBERTa.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())

    return np.vstack(all_preds)


def run_roberta_pipeline(
    train_df,
    val_df,
    test_df,
    model_name="roberta-base",
    batch_size=32,
    max_len=128,
    epochs=5,
    lr=2e-5,
    patience=2,
    seed=42,
):
    """
    Orchestrates the RoBERTa training, validation, and inference process.
    """
    print("\n=== Starting RoBERTa Pipeline ===")
    seed_everything(seed)

    # 1. Prepare Data
    print("Tokenizing and creating DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=batch_size, max_len=max_len
    )

    # 2. Initialize Model
    print(f"Initializing {model_name}...")
    model = ToxicRoBERTa(model_name=model_name, num_classes=len(LABEL_COLS))
    model.to(DEVICE)

    # 3. Optimization Setup
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=lr)
    num_train_steps = int(len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(SUBMISSION_DIR, "best_roberta_model.bin")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Starting training for {epochs} epochs on {DEVICE}...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_roberta_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, criterion
        )
        val_loss, val_auc = evaluate_roberta(model, val_loader, DEVICE, criterion)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val AUC:    {val_auc}")

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved!")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    print("Generating RoBERTa test predictions...")
    test_probs = infer_roberta(model, test_loader, DEVICE)

    return test_probs, best_auc


def generate_submission(test_df, predictions, filename="submission.csv"):
    """
    Saves predictions to a CSV file in the required format.
    """
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    output_path = os.path.join(SUBMISSION_DIR, filename)

    sub_df = pd.DataFrame(predictions, columns=LABEL_COLS)
    sub_df.insert(0, "id", test_df["id"])

    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_full_training(
    load_cached_features=True,
    roberta_epochs=3,
    roberta_batch_size=32,
    ensemble_alpha=0.6,  # Weight for RoBERTa
    seed=42,
):
    """
    Main driver function to run the hybrid ensemble pipeline.
    """
    seed_everything(seed)

    # 1. Load Data
    print("Loading data from metadata...")
    train_df, val_df, test_df = load_data_from_metadata()
    print(
        f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    # 2. NBSVM Pipeline
    nbsvm_preds, nbsvm_auc = train_nbsvm_pipeline(
        train_df, val_df, test_df, load_cached_data=load_cached_features
    )

    # 3. RoBERTa Pipeline
    roberta_preds, roberta_auc = run_roberta_pipeline(
        train_df,
        val_df,
        test_df,
        epochs=roberta_epochs,
        batch_size=roberta_batch_size,
        seed=seed,
    )

    print("\n=== Ensemble Results ===")
    print(f"NBSVM AUC:   {nbsvm_auc}")
    print(f"RoBERTa AUC: {roberta_auc}")

    # 4. Ensemble
    print(
        f"Ensembling predictions (RoBERTa weight: {ensemble_alpha}, NBSVM weight: {1-ensemble_alpha})..."
    )
    final_preds = (ensemble_alpha * roberta_preds) + (
        (1 - ensemble_alpha) * nbsvm_preds
    )

    # 5. Save Submission
    generate_submission(test_df, final_preds)

    return final_preds
