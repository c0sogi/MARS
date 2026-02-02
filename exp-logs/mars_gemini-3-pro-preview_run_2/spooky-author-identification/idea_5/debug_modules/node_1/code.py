import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import library modules
from library.config import Config
from library.utils import seed_everything, load_numpy
from library.data_loader import load_data, get_classical_features, TextDataset
from library.models_classical import run_classical_models
from library.models_transformer import CustomDeberta, CustomRoberta
from library.engine import run_training
from library.ensemble import run_ensemble


def main():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")
    # Override Config constants to ensure quick execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for demo
    Config.NUM_FOLDS = 2  # Only 2 folds for demo
    Config.EPOCHS = 1  # Only 1 epoch for transformer
    Config.BATCH_SIZE = 4  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Re-create directories based on new paths
    Config.create_dirs()
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Folds: {Config.NUM_FOLDS}")

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n[2] Loading Data...")
    # Force reload to ensure we use the debug subset
    # We delete cache files if they exist in the default location to be safe,
    # though Config.WORKING_DIR change handles isolation.
    df_train, df_test = load_data(
        load_cached_data=False,
        debug=Config.DEBUG,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Validation
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    assert "fold" in df_train.columns, "df_train must contain 'fold' column"
    assert "author" in df_train.columns, "df_train must contain 'author' column"
    assert (
        len(df_train) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train data not downsampled correctly"
    assert (
        df_train["fold"].nunique() == Config.NUM_FOLDS
    ), f"Expected {Config.NUM_FOLDS} folds"

    # ------------------------------------------------------------------------
    # 3. Classical Models Pipeline
    # ------------------------------------------------------------------------
    print("\n[3] Running Classical Models Pipeline...")

    # 3.1 Feature Generation
    print("Generating TF-IDF and SVD features...")
    train_texts = df_train["text"].values
    test_texts = df_test["text"].values

    # We pass val_texts as train_texts here just to satisfy the signature for the demo,
    # but in run_classical_models, splitting is handled by indices.
    # Actually, get_classical_features expects (train, val, test).
    # Since we are doing CV inside run_classical_models, we need features for the whole train set.
    # We will pass the full train set as 'train' and 'val' effectively just to get the transformed matrices.
    # A cleaner usage based on the library code:
    train_tfidf, _, test_tfidf, train_svd, _, test_svd = get_classical_features(
        train_texts, train_texts, test_texts
    )

    assert train_tfidf.shape[0] == len(df_train)
    assert train_svd.shape[0] == len(df_train)
    assert test_tfidf.shape[0] == len(df_test)

    # 3.2 Model Training (LR, NB, XGB)
    # We force load_cached_data=False to trigger actual training
    oof_preds_classical, test_preds_classical = run_classical_models(
        df_train,
        df_test,
        train_tfidf,
        test_tfidf,
        train_svd,
        test_svd,
        load_cached_data=False,
    )

    # Validation
    model_keys = ["lr", "nb", "xgb"]
    for key in model_keys:
        assert key in oof_preds_classical
        assert oof_preds_classical[key].shape == (len(df_train), 3)
        assert test_preds_classical[key].shape == (len(df_test), 3)
        # Check probability constraints
        assert np.all(oof_preds_classical[key] >= 0) and np.all(
            oof_preds_classical[key] <= 1
        )

    print("Classical models executed successfully.")

    # ------------------------------------------------------------------------
    # 4. Transformer Models Pipeline (Demonstration)
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating Transformer Pipeline...")

    # 4.1 Dataset & DataLoader
    # Create a small dataset for demonstration
    demo_texts = df_train["text"].iloc[:16].tolist()
    demo_labels = df_train["author"].iloc[:16].tolist()

    ds = TextDataset(demo_texts, demo_labels, tokenizer_name=Config.MODEL_DEBERTA)
    dl = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    batch = next(iter(dl))
    print(f"Batch keys: {batch.keys()}")
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (Config.BATCH_SIZE, Config.MAX_LEN)

    # 4.2 Model Instantiation & Forward Pass
    print(f"Initializing {Config.MODEL_DEBERTA}...")
    # Note: This downloads the model config/weights if not cached.
    # Given the environment, we assume internet access or cached models.
    # We wrap in a try-catch block for the model loading just in case of network issues in the demo environment,
    # but strictly following instructions "All validation checks must fail explicitly", we assume it works.
    device = Config.DEVICE
    model = CustomDeberta(num_classes=3)
    model.to(device)

    # Forward pass check
    with torch.no_grad():
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)

    assert logits.shape == (Config.BATCH_SIZE, 3), "Output shape mismatch for Deberta"
    print("Forward pass successful.")

    # 4.3 Training Loop (Single Fold, Single Epoch)
    print("Running short training loop (1 epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Use the same dataloader for train and val for this quick demo
    best_loss, best_preds = run_training(
        model=model,
        train_dataloader=dl,
        val_dataloader=dl,
        optimizer=optimizer,
        device=device,
        num_epochs=Config.EPOCHS,
        patience=1,
        fold=0,
        model_name="demo_deberta",
    )

    assert best_preds is not None
    assert best_preds.shape == (16, 3)
    print("Transformer training loop executed successfully.")

    # ------------------------------------------------------------------------
    # 5. Ensemble Pipeline
    # ------------------------------------------------------------------------
    print("\n[5] Running Ensemble Pipeline...")

    # Prepare inputs for ensemble
    # We use the classical OOF predictions we generated earlier.
    # For the purpose of this demo, we won't include the transformer predictions
    # in the ensemble input because we didn't generate full OOF predictions for it
    # (we only trained on a tiny subset of fold 0).

    y_true_map = {"EAP": 0, "HPL": 1, "MWS": 2}
    y_true = df_train["author"].map(y_true_map).values
    test_ids = df_test["id"].values

    # Run ensemble
    final_probs = run_ensemble(
        oof_preds=oof_preds_classical,
        test_preds=test_preds_classical,
        y_true=y_true,
        test_ids=test_ids,
    )

    # Validation
    assert final_probs.shape == (len(df_test), 3)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == ["id", "EAP", "HPL", "MWS"]
    assert len(df_sub) == len(df_test)

    print(f"Ensemble complete. Submission saved to {submission_path}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
