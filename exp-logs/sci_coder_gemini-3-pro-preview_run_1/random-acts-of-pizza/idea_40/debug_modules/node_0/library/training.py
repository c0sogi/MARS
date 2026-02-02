import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library import config, utils, dataset, mlp_model
from library.features import FeatureEngineer


def train_mlp_model(load_cached_data=True, debug=config.DEBUG):
    """
    Orchestrates the training, validation, and inference of the SkipGatedDualQueryMLP model.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from cache.
        debug (bool): Whether to run in debug mode with a data subset.

    Returns:
        dict: A dictionary containing:
            - 'model': The trained PyTorch model.
            - 'val_auc': The best validation ROC-AUC score.
            - 'test_preds': Numpy array of probabilities for the test set.
            - 'request_ids': Numpy array of request IDs for the test set.
    """
    # 1. Setup
    utils.set_seed()

    # Determine device
    device_name = config.MLP_PARAMS.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"MLP Training using device: {device}")

    # 2. Load Data
    print("Loading datasets for MLP...")
    train_dataset, val_dataset, test_dataset = dataset.get_mlp_datasets(
        load_cached_data=load_cached_data, debug=debug
    )

    batch_size = config.MLP_PARAMS["batch_size"]
    # num_workers=0 ensures compatibility and reproducibility without complex setup
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 3. Initialize Model
    # Determine input dimension for metadata from the dataset
    # We grab the first sample to check the shape of the 'meta' tensor
    first_sample = train_dataset[0]
    input_meta_dim = first_sample["meta"].shape[0]

    print(f"Initializing SkipGatedDualQueryMLP with metadata dim: {input_meta_dim}")
    model = mlp_model.SkipGatedDualQueryMLP(input_meta_dim=input_meta_dim)
    model.to(device)

    # 4. Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.MLP_PARAMS["learning_rate"],
        weight_decay=config.MLP_PARAMS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    epochs = config.MLP_PARAMS["epochs"]
    patience = config.MLP_PARAMS["patience"]

    best_val_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(config.CACHE_DIR, "best_mlp_model.pth")

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            # Move data to device
            title_emb = batch["title_emb"].to(device)
            body_emb = batch["body_emb"].to(device)
            hist_seq = batch["hist_seq"].to(device)
            hist_mask = batch["hist_mask"].to(device)
            meta = batch["meta"].to(device)
            cons = batch["cons"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(title_emb, body_emb, hist_seq, hist_mask, meta, cons)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_samples += labels.size(0)

        avg_train_loss = train_loss_sum / train_samples if train_samples > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                title_emb = batch["title_emb"].to(device)
                body_emb = batch["body_emb"].to(device)
                hist_seq = batch["hist_seq"].to(device)
                hist_mask = batch["hist_mask"].to(device)
                meta = batch["meta"].to(device)
                cons = batch["cons"].to(device)
                labels = batch["label"].to(device)

                logits = model(title_emb, body_emb, hist_seq, hist_mask, meta, cons)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_samples += labels.size(0)

                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_sum / val_samples if val_samples > 0 else 0.0

        # Calculate AUC
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            # Handle edge case in debugging with single class
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_val_auc}"
                )
                break

    # 6. Load Best Model
    if os.path.exists(best_model_path):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No model file saved (training might have failed or 0 epochs).")

    # 7. Inference on Test Set
    print("Generating predictions on test set...")
    model.eval()
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            title_emb = batch["title_emb"].to(device)
            body_emb = batch["body_emb"].to(device)
            hist_seq = batch["hist_seq"].to(device)
            hist_mask = batch["hist_mask"].to(device)
            meta = batch["meta"].to(device)
            cons = batch["cons"].to(device)

            logits = model(title_emb, body_emb, hist_seq, hist_mask, meta, cons)
            probs = torch.sigmoid(logits)
            test_preds.extend(probs.cpu().numpy())

    test_preds = np.array(test_preds)

    # 8. Retrieve Request IDs
    # We need to access the FeatureEngineer cache or re-process to get the IDs corresponding to test set
    # Since dataset.get_mlp_datasets handles the internal logic, we use the FeatureEngineer directly here
    # to ensure we get the IDs aligned with the test set.
    # Note: We must ensure the debug flag matches what was used to create the dataset.
    original_debug = config.DEBUG
    config.DEBUG = debug
    try:
        fe = FeatureEngineer()
        _, mlp_out = fe.process_data(load_cached_data=True)
        request_ids = mlp_out["request_ids_test"]
    finally:
        config.DEBUG = original_debug

    return {
        "model": model,
        "val_auc": best_val_auc,
        "test_preds": test_preds,
        "request_ids": request_ids,
    }
