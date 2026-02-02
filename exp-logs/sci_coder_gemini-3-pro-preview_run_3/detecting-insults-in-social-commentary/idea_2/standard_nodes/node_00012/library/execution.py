import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import roc_auc_score

from library.utils import load_data, save_submission, set_seed
from library.data_factory import InsultDataset
from library.modeling import RoBERTaClassifier


def run_neural(
    load_cached_data=True,
    max_samples=None,
    epochs=4,
    batch_size=16,
    lr=1e-5,
    device=None,
    model_name="roberta-large",
):
    """
    Executes the Neural Stream (RoBERTa).

    Args:
        load_cached_data (bool): Whether to use cached data.
        max_samples (int, optional): Truncate training data for debugging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        lr (float): Learning rate.
        device (torch.device, optional): Device to run on.
        model_name (str): Name of the pre-trained model.

    Returns:
        tuple: (val_preds, test_preds) as numpy arrays.
    """
    print("--- Starting Neural Stream ---")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Stream using device: {device}")

    # Load Data
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # Debugging: Truncate training data
    if max_samples is not None and max_samples < len(train_df):
        print(f"Neural: Truncating training data to {max_samples} samples.")
        train_df = train_df.iloc[:max_samples].reset_index(drop=True)

    # Initialize Tokenizer
    print(f"Initializing Tokenizer ({model_name})...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Create Datasets
    # Increase max_len slightly to capture more context
    train_dataset = InsultDataset(train_df, tokenizer, max_len=160)
    val_dataset = InsultDataset(val_df, tokenizer, max_len=160)
    test_dataset = InsultDataset(test_df, tokenizer, max_len=160)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model
    print(f"Initializing RoBERTa Model ({model_name})...")
    # Cite solution_lesson_node_00010: Increased dropout to 0.2 and freezing bottom 6 layers.
    model = RoBERTaClassifier(model_name=model_name, dropout=0.2, freeze_layers=6)

    # Train Model
    # train_model handles the training loop and early stopping, and loads best weights
    model.train_model(train_loader, val_loader, device, epochs=epochs, lr=lr)

    # Inference
    print("Generating Neural predictions...")
    val_preds = model.predict(val_loader, device)
    test_preds = model.predict(test_loader, device)

    # Evaluation
    y_val = val_df["Insult"].values
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Neural Validation AUC: {val_auc}")

    return val_preds, test_preds


def run_pipeline(load_cached_data=True, max_samples=None, epochs=4, batch_size=16):
    """
    Main execution pipeline:
    1. Train Neural Model (RoBERTa-Large)
    2. Generate Submission
    """
    set_seed(42)

    # Run Neural Stream
    # Cite solution_lesson_node_00006: Removed NBSVM stream as it provided no marginal gain.
    neural_val_preds, neural_test_preds = run_neural(
        load_cached_data=load_cached_data,
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-5,
        model_name="roberta-large",
    )

    val_df = load_data("val", load_cached_data=load_cached_data)
    y_val = val_df["Insult"].values
    val_auc = roc_auc_score(y_val, neural_val_preds)
    print(f"Validation AUC: {val_auc}")

    # Save Submission
    print("\n--- Saving Submission ---")
    test_df = load_data("test", load_cached_data=load_cached_data)
    save_submission(neural_test_preds, test_df, output_dir="./submission")
    print("Submission saved to ./submission/submission.csv")
