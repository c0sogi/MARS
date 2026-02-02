import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error

from library.config import (
    DEVICE,
    BATCH_SIZE,
    LR,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    SEED,
)
from library.utils import seed_everything
from library.data_processing import load_notebook_data
from library.feature_extraction import SparseVectorizer, DenseInputProcessor
from library.model_definitions import RidgeRegressorWrapper, TransformerRegressor

seed_everything(SEED)


class MarkdownDataset(Dataset):
    """
    PyTorch Dataset for loading markdown cells and their structural context.
    """

    def __init__(self, df):
        self.texts = df["text"].values.astype(str)
        self.contexts = df["context"].values.astype(str)
        self.ranks = df["rank"].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.contexts[idx], self.ranks[idx]


def create_collate_fn(processor):
    """
    Creates a collate function that uses the DenseInputProcessor to tokenize batches.
    """

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        contexts = [item[1] for item in batch]
        ranks = [item[2] for item in batch]

        # Tokenize batch
        inputs = processor.process_batch(texts, contexts)

        # Convert targets to tensor
        targets = torch.tensor(ranks, dtype=torch.float32)

        return inputs, targets

    return collate_fn


def train_sparse_model(load_cached_data=True):
    """
    Trains the Sparse Ridge Regression model.
    Fits TF-IDF on the combined corpus (Train + Val) for better vocabulary coverage,
    then trains Ridge on the Train split and evaluates on Val.
    """
    print("--- Starting Sparse Model Training ---")

    # Load Data
    df_train = load_notebook_data("train", load_cached_data=load_cached_data)
    df_val = load_notebook_data("val", load_cached_data=load_cached_data)

    # Prepare Corpus for Vectorizer (Fit on Train + Val to capture full vocab)
    print("Fitting Vectorizer on combined corpus...")
    full_corpus = pd.concat([df_train["text"], df_val["text"]], axis=0).astype(str)

    vectorizer = SparseVectorizer()
    vectorizer.fit(full_corpus)
    vectorizer.save()

    # Transform Data
    print("Transforming training data...")
    X_train = vectorizer.transform(df_train["text"].astype(str))
    y_train = df_train["rank"].values

    print("Transforming validation data...")
    X_val = vectorizer.transform(df_val["text"].astype(str))
    y_val = df_val["rank"].values

    # Train Ridge Model
    print("Fitting Ridge Regressor...")
    ridge_model = RidgeRegressorWrapper()
    ridge_model.fit(X_train, y_train)
    ridge_model.save()

    # Evaluation
    print("Evaluating Sparse Model...")
    preds_val = ridge_model.predict(X_val)
    mse = mean_squared_error(y_val, preds_val)
    print(f"Sparse Model Validation MSE: {mse}")

    return ridge_model


def train_dense_model(load_cached_data=True):
    """
    Trains the Dense Transformer Model using Mixed Precision (FP16).
    """
    print("--- Starting Dense Model Training ---")

    # Load Data
    df_train = load_notebook_data("train", load_cached_data=load_cached_data)
    df_val = load_notebook_data("val", load_cached_data=load_cached_data)

    # Initialize Processor
    processor = DenseInputProcessor()

    # Create Datasets and DataLoaders
    train_dataset = MarkdownDataset(df_train)
    val_dataset = MarkdownDataset(df_val)

    collate_fn = create_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Initialize Model
    model = TransformerRegressor()
    model.to(DEVICE)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    num_training_steps = len(train_loader) * NUM_EPOCHS
    # 10% warmup
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss Function
    criterion = nn.MSELoss()

    # Mixed Precision Scaler
    scaler = GradScaler()

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_accum = 0.0

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for step, (inputs, targets) in enumerate(train_loader):
            # Move to device
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward Pass with Mixed Precision
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss = loss / ACCUMULATION_STEPS

            # Backward Pass
            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                # Optimizer Step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss_accum += loss.item() * ACCUMULATION_STEPS

        avg_train_loss = train_loss_accum / len(train_loader)
        print(f"Average Training MSE: {avg_train_loss}")

        # Validation Loop
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                targets = targets.to(DEVICE)

                with autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, targets)

                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        print(f"Validation MSE: {avg_val_loss}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save()
            print("New best model saved.")

    return model
