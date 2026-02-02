import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from library.config import RF_PARAMS, MLP_PARAMS, DEVICE, NUM_WORKERS, WORKING_DIR
from library.utils import seed_everything
from library.dataset import PizzaDataset
from library.model_mlp import CredibilityGatedMLP


def evaluate_preds(y_true, y_pred):
    """
    Calculates and returns the ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def train_rf(data_rf):
    """
    Trains the Random Forest model (Stream A).

    Args:
        data_rf (dict): Dictionary containing X_train, y_train, X_val, y_val, X_test.

    Returns:
        dict: Dictionary containing predictions for 'train', 'val', and 'test'.
        model: The trained RandomForestClassifier.
    """
    print("\n=== Training Random Forest (Stream A) ===")
    seed_everything(RF_PARAMS["random_state"])

    # Initialize model
    rf_model = RandomForestClassifier(**RF_PARAMS)

    # Train
    print(
        f"Fitting Random Forest on {data_rf['X_train'].shape[0]} samples with {data_rf['X_train'].shape[1]} features..."
    )
    rf_model.fit(data_rf["X_train"], data_rf["y_train"])

    # Predict
    print("Generating predictions...")
    # predict_proba returns [prob_0, prob_1], we want prob_1
    train_preds = rf_model.predict_proba(data_rf["X_train"])[:, 1]
    val_preds = rf_model.predict_proba(data_rf["X_val"])[:, 1]
    test_preds = rf_model.predict_proba(data_rf["X_test"])[:, 1]

    # Evaluate
    train_auc = evaluate_preds(data_rf["y_train"], train_preds)
    val_auc = evaluate_preds(data_rf["y_val"], val_preds)

    print(f"Random Forest Train AUC: {train_auc}")
    print(f"Random Forest Val AUC:   {val_auc}")

    predictions = {"train": train_preds, "val": val_preds, "test": test_preds}

    return predictions, rf_model


def train_mlp(data_mlp):
    """
    Trains the Credibility-Gated MLP (Stream B).

    Args:
        data_mlp (dict): Dictionary containing 'train', 'val', 'test' sub-dictionaries.
                         Each sub-dict has 'req_emb', 'hist_emb', 'meta', and optionally 'y'.

    Returns:
        dict: Dictionary containing predictions for 'train', 'val', and 'test'.
        model: The trained CredibilityGatedMLP (best state loaded).
    """
    print("\n=== Training Credibility-Gated MLP (Stream B) ===")
    seed_everything(42)  # Ensure reproducibility for NN initialization

    # 1. Prepare DataLoaders
    batch_size = MLP_PARAMS["batch_size"]

    train_dataset = PizzaDataset(data_mlp["train"])
    val_dataset = PizzaDataset(data_mlp["val"])
    test_dataset = PizzaDataset(data_mlp["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # 2. Initialize Model
    # Determine input dimension for metadata from the data
    input_dim_meta = data_mlp["train"]["meta"].shape[1]

    model = CredibilityGatedMLP(
        input_dim_meta=input_dim_meta,
        embedding_dim=MLP_PARAMS["embedding_dim"],
        hidden_dim=MLP_PARAMS["hidden_dim"],
        dropout_rate=MLP_PARAMS["dropout_rate"],
    ).to(DEVICE)

    # 3. Setup Training Components
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MLP_PARAMS["learning_rate"],
        weight_decay=MLP_PARAMS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop with Early Stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "mlp_best_model.pth")

    print(f"Starting training on device: {DEVICE}")

    for epoch in range(MLP_PARAMS["num_epochs"]):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        all_train_preds = []
        all_train_targets = []

        for batch in train_loader:
            # Move data to device
            req_emb = batch["request_emb"].to(DEVICE)
            hist_emb = batch["history_emb"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE).unsqueeze(1)  # (Batch, 1)

            optimizer.zero_grad()

            # Forward pass
            logits = model(req_emb, hist_emb, meta, mask)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss_accum += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_train_preds.extend(probs)
            all_train_targets.extend(labels.cpu().numpy())

        avg_train_loss = train_loss_accum / len(train_loader)
        train_auc = evaluate_preds(
            np.array(all_train_targets), np.array(all_train_preds)
        )

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                req_emb = batch["request_emb"].to(DEVICE)
                hist_emb = batch["history_emb"].to(DEVICE)
                meta = batch["metadata"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE).unsqueeze(1)

                logits = model(req_emb, hist_emb, meta, mask)
                loss = criterion(logits, labels)

                val_loss_accum += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_val_preds.extend(probs)
                all_val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)
        val_auc = evaluate_preds(np.array(all_val_targets), np.array(all_val_preds))

        print(
            f"Epoch {epoch+1}/{MLP_PARAMS['num_epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc} | "
            f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc}"
        )

        # --- Early Stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print("  -> New best model saved!")
        else:
            patience_counter += 1
            # print(f"  -> No improvement. Patience: {patience_counter}/{MLP_PARAMS['patience']}")

        if patience_counter >= MLP_PARAMS["patience"]:
            print("Early stopping triggered.")
            break

    # 5. Load Best Model and Generate Final Predictions
    print(f"Loading best model with Val AUC: {best_val_auc}")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    def get_preds(loader):
        preds = []
        with torch.no_grad():
            for batch in loader:
                req_emb = batch["request_emb"].to(DEVICE)
                hist_emb = batch["history_emb"].to(DEVICE)
                meta = batch["metadata"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                logits = model(req_emb, hist_emb, meta, mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
        return np.array(preds).flatten()

    final_train_preds = get_preds(train_loader)
    final_val_preds = get_preds(val_loader)
    final_test_preds = get_preds(test_loader)

    predictions = {
        "train": final_train_preds,
        "val": final_val_preds,
        "test": final_test_preds,
    }

    return predictions, model
