import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_cosine_schedule_with_warmup

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import ToxicDataset, get_nbsvm_features
import library.dataset  # For monkey-patching
from library.nbsvm_model import NBSVM, train_and_predict_nbsvm
from library.neural_model import DebertaClassifier
from library.trainer import train_fn, inference_fn


def demo_main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup & Configuration Override
    # We use a tiny model and a small batch size for the demo to finish in seconds.
    print("\n[Step 1] Configuring environment...")
    Config.MODEL_NAME = "prajjwal1/bert-tiny"
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"

    # Update cache paths to use the demo directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.CACHE_NBSVM_WORD_TRAIN = os.path.join(
        Config.WORKING_DIR, "nbsvm_word_train.npz"
    )
    Config.CACHE_NBSVM_WORD_TEST = os.path.join(
        Config.WORKING_DIR, "nbsvm_word_test.npz"
    )
    Config.CACHE_NBSVM_CHAR_TRAIN = os.path.join(
        Config.WORKING_DIR, "nbsvm_char_train.npz"
    )
    Config.CACHE_NBSVM_CHAR_TEST = os.path.join(
        Config.WORKING_DIR, "nbsvm_char_test.npz"
    )

    seed_everything(Config.SEED)
    print("Configuration updated for speed.")

    # 2. Data Subsampling Strategy
    # We need a subset that contains at least a few positives for every class
    # so NBSVM/LogisticRegression doesn't crash.
    print("\n[Step 2] Preparing data subset...")

    def get_stratified_subset_indices(meta_path, n_per_class=5, n_negatives=50):
        df = pd.read_csv(meta_path)
        indices = set()

        # Add some negatives
        negatives = df[df[Config.LABEL_COLS].sum(axis=1) == 0]
        if len(negatives) > 0:
            indices.update(
                negatives.sample(
                    min(n_negatives, len(negatives)), random_state=Config.SEED
                ).index.tolist()
            )

        # Add positives for each class
        for col in Config.LABEL_COLS:
            positives = df[df[col] == 1]
            if len(positives) > 0:
                indices.update(
                    positives.sample(
                        min(n_per_class, len(positives)), random_state=Config.SEED
                    ).index.tolist()
                )

        return sorted(list(indices))

    # Define a mock loader that returns small dataframes
    def mock_load_data_splits():
        print("  -> Loading subset of data via mock loader...")
        # Get indices
        train_idxs = get_stratified_subset_indices(Config.TRAIN_META_CSV)
        val_idxs = get_stratified_subset_indices(
            Config.VAL_META_CSV, n_per_class=2, n_negatives=20
        )
        # Test just takes first 50
        test_idxs = list(range(50))

        # Helper to load and slice
        def load_slice(csv_path, meta_path, indices, is_test=False):
            meta = pd.read_csv(meta_path)
            raw = pd.read_csv(csv_path)

            # Filter metadata by our selected indices
            # Note: The indices in `indices` correspond to rows in the metadata file
            meta_subset = meta.iloc[indices].reset_index(drop=True)

            # Map to raw rows
            source_rows = meta_subset["source_row_index"].values
            data_subset = raw.iloc[source_rows].reset_index(drop=True)

            # Merge logic similar to original library
            if not is_test:
                data_subset = data_subset.drop(
                    columns=Config.LABEL_COLS, errors="ignore"
                )
                data_subset = data_subset.drop(columns=["id"], errors="ignore")
                combined = pd.concat(
                    [meta_subset, data_subset[["comment_text"]]], axis=1
                )
            else:
                data_subset = data_subset.drop(columns=["id"], errors="ignore")
                combined = pd.concat(
                    [meta_subset, data_subset[["comment_text"]]], axis=1
                )

            return combined

        df_train = load_slice(Config.TRAIN_CSV, Config.TRAIN_META_CSV, train_idxs)
        df_val = load_slice(Config.TRAIN_CSV, Config.VAL_META_CSV, val_idxs)
        df_test = load_slice(
            Config.TEST_CSV, Config.TEST_META_CSV, test_idxs, is_test=True
        )

        print(
            f"  -> Subset shapes: Train={df_train.shape}, Val={df_val.shape}, Test={df_test.shape}"
        )
        return df_train, df_val, df_test

    # Monkey-patch the library function
    library.dataset.load_data_splits = mock_load_data_splits

    # Load the data
    df_train, df_val, df_test = library.dataset.load_data_splits()

    # 3. NBSVM Demonstration
    print("\n[Step 3] Demonstrating NBSVM Pipeline...")

    # Generate features
    # We force re-computation (load_cached_data=False) to test the generation logic
    X_train, X_val, X_test = get_nbsvm_features(
        df_train, df_val, df_test, load_cached_data=False
    )

    assert X_train.shape[0] == len(df_train), "NBSVM Feature rows mismatch train set"
    assert X_val.shape[0] == len(df_val), "NBSVM Feature rows mismatch val set"

    # Train and Predict
    val_preds_nb, test_preds_nb = train_and_predict_nbsvm(
        X_train,
        df_train[Config.LABEL_COLS].values,
        X_val,
        df_val[Config.LABEL_COLS].values,
        X_test,
    )

    # Validation
    assert val_preds_nb.shape == (
        len(df_val),
        Config.NUM_LABELS,
    ), "NBSVM Val prediction shape incorrect"
    assert test_preds_nb.shape == (
        len(df_test),
        Config.NUM_LABELS,
    ), "NBSVM Test prediction shape incorrect"

    auc_score = calculate_roc_auc(df_val[Config.LABEL_COLS].values, val_preds_nb)
    print(f"  -> NBSVM Demo AUC: {auc_score:.4f}")
    assert 0.0 <= auc_score <= 1.0, "AUC score out of bounds"

    # 4. Neural Model Demonstration
    print("\n[Step 4] Demonstrating Neural Pipeline (DeBERTa/BERT)...")

    device = Config.DEVICE
    print(f"  -> Device: {device}")

    # Dataset & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_dataset = ToxicDataset(df_train, tokenizer, Config.MAX_LEN)
    val_dataset = ToxicDataset(df_val, tokenizer, Config.MAX_LEN)

    # Check dataset item
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "labels" in sample_item
    assert sample_item["labels"].shape[0] == Config.NUM_LABELS

    train_loader = DataLoader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False
    )

    # Model Initialization
    model = DebertaClassifier(Config.MODEL_NAME, Config.NUM_LABELS)
    model.to(device)

    # Optimizer Setup (Testing get_optimizer_params)
    optimizer_params = model.get_optimizer_params(
        base_lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )
    optimizer = AdamW(optimizer_params, lr=Config.LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 10)  # Dummy steps

    # Training Loop (1 Epoch)
    print("  -> Running training step (train_fn)...")
    loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch=0)
    print(f"  -> Train Loss: {loss:.4f}")
    assert loss > 0, "Training loss should be positive"

    # Inference
    print("  -> Running inference step (inference_fn)...")
    val_preds_nn = inference_fn(model, val_loader, device)

    # Validation
    assert val_preds_nn.shape == (
        len(df_val),
        Config.NUM_LABELS,
    ), "Neural Val prediction shape incorrect"
    auc_score_nn = calculate_roc_auc(df_val[Config.LABEL_COLS].values, val_preds_nn)
    print(f"  -> Neural Demo AUC: {auc_score_nn:.4f}")

    print("\n=== Demonstration Complete: All components verified successfully. ===")


if __name__ == "__main__":
    demo_main()
