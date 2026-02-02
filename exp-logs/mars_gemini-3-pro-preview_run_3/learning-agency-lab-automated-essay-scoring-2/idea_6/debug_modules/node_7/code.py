import os
import shutil
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from sklearn.model_selection import StratifiedKFold

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, compute_qwk, get_logger
from library.data import process_data, get_dataloaders
from library.model import EssayModel
import library.engine
import importlib

importlib.reload(library.engine)
from library.engine import (
    get_optimizer_params,
    train_fn,
    valid_fn,
    extract_embeddings_fn,
)
from library.stacking import create_stacking_dataset, train_lgbm_and_predict
import library.stacking  # Imported to mock load_meta_features


def run_pipeline_demo():
    print("=== Essay Scoring Pipeline Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")
    # Override CFG settings to ensure the script completes quickly (< 1 hour)
    CFG.debug = True
    CFG.debug_subset_size = 50  # Use only 50 samples
    CFG.epochs = 1  # Train for only 1 epoch
    CFG.batch_size = 2  # Small batch size
    CFG.n_folds = 2  # Use 2 folds for stacking CV
    CFG.print_freq = 5  # Frequent printing
    CFG.accum_iter = 1  # No gradient accumulation
    CFG.use_awp = True  # Enable AWP to verify it works
    CFG.awp_start_epoch = 0  # Start AWP immediately

    # Clean up any previous runs in working directory
    if os.path.exists(CFG.cache_dir):
        shutil.rmtree(CFG.cache_dir)
    os.makedirs(CFG.cache_dir, exist_ok=True)

    seed_everything(CFG.seed)
    device = CFG.device
    print(f"Running on device: {device}")

    # ------------------------------------------------------------------------
    # 2. Data Processing and Loading
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Processing...")

    # Generate DataLoaders (this internally calls process_data and caches it)
    # We use Fold 0 for this demonstration
    train_loader, val_loader, test_loader = get_dataloaders(
        fold=0, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    chunk_mask = batch["chunk_mask"]
    meta_features = batch["meta_features"]
    labels = batch["labels"]

    print(f"Input IDs Shape: {input_ids.shape}")  # Expected: (B, n_chunks, seq_len)
    print(f"Meta Features Shape: {meta_features.shape}")  # Expected: (B, 4)
    print(f"Labels Shape: {labels.shape}")  # Expected: (B,)

    # Logic Verification
    assert input_ids.dim() == 3, "Input IDs must be 3D tensor (Batch, Chunks, SeqLen)"
    assert meta_features.shape[1] == 4, "Meta features must have 4 columns"
    assert labels.dtype == torch.float32, "Labels must be float32"

    # ------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = EssayModel(pretrained=True).to(device)

    # Move batch to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    chunk_mask = chunk_mask.to(device)

    # Perform Forward Pass
    with torch.no_grad():
        output = model(input_ids, attention_mask, chunk_mask)

    logits = output["logits"]
    embedding = output["embedding"]

    print(f"Logits Shape: {logits.shape}")
    print(f"Embedding Shape: {embedding.shape}")

    # Logic Verification
    assert logits.shape == (CFG.batch_size, 1), "Logits shape mismatch"
    # DeBERTa-v3-large hidden size is 1024
    assert embedding.shape == (CFG.batch_size, 1024), "Embedding shape mismatch"

    # ------------------------------------------------------------------------
    # 4. Training Loop (Engine)
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Training Loop...")

    # Setup Optimizer and Scheduler
    optimizer_params = get_optimizer_params(
        model, encoder_lr=CFG.encoder_lr, decoder_lr=CFG.head_lr
    )
    optimizer = AdamW(
        optimizer_params, lr=CFG.encoder_lr, weight_decay=CFG.weight_decay
    )
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=len(train_loader)
    )
    criterion = nn.MSELoss()

    # Run 1 Epoch of Training
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
    )

    print(f"Epoch 0 Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss returned NaN"

    # ------------------------------------------------------------------------
    # 5. Validation Loop
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Validation Loop...")

    val_loss, val_preds = valid_fn(val_loader, model, criterion, device)

    print(f"Validation Loss: {val_loss:.4f}")
    print(
        f"Validation Predictions Range: [{val_preds.min():.2f}, {val_preds.max():.2f}]"
    )

    assert len(val_preds) == len(val_loader.dataset), "Prediction count mismatch"

    # ------------------------------------------------------------------------
    # 6. Stacking Integration
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Stacking Pipeline...")

    # Extract embeddings from Validation set (Simulating OOF) and Test set
    print("Extracting embeddings...")
    oof_embeddings, oof_labels = extract_embeddings_fn(val_loader, model, device)
    test_embeddings, _ = extract_embeddings_fn(test_loader, model, device)

    print(f"OOF Embeddings Shape: {oof_embeddings.shape}")

    # Mock load_meta_features to return the correct subset of data
    # The original function loads the full dataset from parquet, which would cause
    # a shape mismatch with our debug subset embeddings.
    def mock_load_meta_features(is_train=True, load_cached_data=True):
        cache_name = "train_processed" if is_train else "test_processed"
        df = pd.read_parquet(os.path.join(CFG.cache_dir, f"{cache_name}.parquet"))

        meta_cols = ["char_count", "word_count", "sentence_count", "unique_word_ratio"]

        if is_train:
            # Replicate the Fold 0 Validation split logic from get_dataloaders
            skf = StratifiedKFold(
                n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed
            )
            df["fold"] = -1
            for f, (t_idx, v_idx) in enumerate(skf.split(df, df["score"])):
                df.loc[v_idx, "fold"] = f

            # Select Fold 0
            subset_df = df[df["fold"] == 0].reset_index(drop=True)
            # Apply debug slicing
            if CFG.debug:
                subset_df = subset_df.head(CFG.debug_subset_size)
            return subset_df[meta_cols].values.astype(np.float32), subset_df
        else:
            return df[meta_cols].values.astype(np.float32), df

    # Apply Mock
    library.stacking.load_meta_features = mock_load_meta_features

    # Run LightGBM Stacking
    # This trains LGBM on the "OOF" embeddings + meta features and predicts on Test
    qwk_score = train_lgbm_and_predict(
        oof_embeddings=oof_embeddings,
        train_labels=oof_labels,
        test_embeddings=test_embeddings,
        load_cached_data=False,  # Force recreation of stacking features
    )

    print(f"Stacking QWK Score (on debug subset): {qwk_score:.4f}")

    # Verify Submission File
    submission_path = os.path.join(CFG.submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
