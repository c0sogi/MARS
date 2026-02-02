import os
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, logging as transformers_logging

# --- Configuration & Setup ---
# Suppress unnecessary warnings and logs for a clean output
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library components
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.features import MechanicsFeatureExtractor
from library.dataset import get_dataloaders, get_test_dataloader
from library.model_semantic import (
    DebertaV3Regressor,
    train_one_epoch,
    validate_one_epoch,
    predict_semantic,
)
from library.model_lexical import LexicalRegressor
from library.meta_learner import MetaLearner, optimize_thresholds, apply_thresholds


def run_demonstration():
    print("=== Starting Essay Scoring System Demonstration ===\n")

    # 1. Configure for Speed (Demo Mode)
    # We override Config attributes to ensure the script finishes quickly.
    print("[Setup] Configuring environment for rapid execution...")
    seed_everything(42)

    Config.set_debug_mode(True)  # Sets EPOCHS=1, reduces estimators
    Config.MAX_LENGTH = 128  # Reduce sequence length for speed
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.LGB_PARAMS["n_estimators"] = 20
    Config.LGB_PARAMS["verbose"] = -1

    # Setup isolated working directory for this run
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    Config.WORKING_DIR = demo_dir
    Config.MODEL_DIR = os.path.join(demo_dir, "models")
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.setup()

    # 2. Load and Subsample Data
    # We use a tiny subset (20 train, 10 val) to verify logic without waiting.
    print("[Data] Loading and subsampling metadata...")
    df_train_full = pd.read_csv("./metadata/train.csv")
    df_val_full = pd.read_csv("./metadata/val.csv")

    df_train = df_train_full.head(20).reset_index(drop=True)
    df_val = df_val_full.head(10).reset_index(drop=True)

    print(f"   Train subset: {df_train.shape}")
    print(f"   Val subset:   {df_val.shape}")

    # 3. Mechanics Branch
    print("\n[Branch 1/3] Mechanics Feature Extraction...")
    mech_extractor = MechanicsFeatureExtractor()

    # Extract features (force recompute by disabling cache loading)
    mech_train = mech_extractor.extract_features(
        df_train, "train_demo", load_cached_data=False
    )
    mech_val = mech_extractor.extract_features(
        df_val, "val_demo", load_cached_data=False
    )

    # Verify
    assert mech_train.shape == (len(df_train), len(Config.MECHANICS_FEATURES))
    assert mech_val.shape == (len(df_val), len(Config.MECHANICS_FEATURES))
    assert not mech_train.isna().any().any(), "Mechanics features contain NaNs"
    print("   Mechanics features extracted successfully.")

    # 4. Lexical Branch
    print("\n[Branch 2/3] Lexical Model (TF-IDF + Ridge)...")
    lex_model = LexicalRegressor()

    # Train
    lex_model.fit(df_train["full_text"], df_train["score"])

    # Predict
    lex_train_preds = lex_model.predict(df_train["full_text"])
    lex_val_preds = lex_model.predict(df_val["full_text"])

    # Verify Save/Load
    lex_path = os.path.join(Config.MODEL_DIR, "lexical_demo.joblib")
    lex_model.save(lex_path)
    loaded_lex = LexicalRegressor().load(lex_path)
    loaded_preds = loaded_lex.predict(df_val["full_text"])

    assert np.allclose(lex_val_preds, loaded_preds), "Lexical model persistence failed"
    print(
        f"   Lexical Validation QWK: {quadratic_weighted_kappa(df_val['score'], lex_val_preds):.4f}"
    )

    # 5. Semantic Branch
    print("\n[Branch 3/3] Semantic Model (DeBERTa)...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_BACKBONE)

    # Create DataLoaders
    train_loader, val_loader = get_dataloaders(
        df_train, df_val, tokenizer, load_cached_data=False
    )

    # Initialize Model & Optimizer
    sem_model = DebertaV3Regressor()
    sem_model.to(Config.DEVICE)
    optimizer = torch.optim.AdamW(sem_model.parameters(), lr=1e-5)
    criterion = torch.nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Train (1 Epoch)
    print("   Training Semantic Model...")
    train_loss = train_one_epoch(
        sem_model,
        train_loader,
        optimizer,
        None,
        criterion,
        Config.DEVICE,
        0,
        awp=None,
        scaler=scaler,
    )

    # Validate
    val_loss, sem_val_preds, val_qwk = validate_one_epoch(
        sem_model, val_loader, criterion, Config.DEVICE
    )
    print(f"   Semantic Train Loss: {train_loss:.4f} | Val QWK: {val_qwk:.4f}")

    # Generate predictions for Train set (for Meta-Learner)
    # We use get_test_dataloader to get a non-shuffled loader for the training set
    train_infer_loader = get_test_dataloader(df_train, tokenizer, load_cached_data=True)
    sem_train_preds = predict_semantic(sem_model, train_infer_loader, Config.DEVICE)

    assert len(sem_train_preds) == len(df_train)
    assert len(sem_val_preds) == len(df_val)

    # 6. Meta-Learner (Stacking)
    print("\n[Meta-Learner] Stacking & Threshold Optimization...")

    # Construct Feature Matrix: Mechanics + Lexical Preds + Semantic Preds
    # (Note: In production, train preds should be OOF, here we use in-sample for demo)
    X_train = mech_train.copy()
    X_train["lex_pred"] = lex_train_preds
    X_train["sem_pred"] = sem_train_preds

    X_val = mech_val.copy()
    X_val["lex_pred"] = lex_val_preds
    X_val["sem_pred"] = sem_val_preds

    y_train = df_train["score"].values
    y_val = df_val["score"].values

    # Train LightGBM Meta-Learner
    meta = MetaLearner()
    meta.fit(X_train, y_train, X_val, y_val)

    # Generate Continuous Predictions
    final_preds_continuous = meta.predict(X_val)

    # Optimize Thresholds
    best_thresholds = optimize_thresholds(y_val, final_preds_continuous)

    # Apply Thresholds to get Integers
    final_preds_int = apply_thresholds(final_preds_continuous, best_thresholds)

    # Final Evaluation
    final_qwk = quadratic_weighted_kappa(y_val, final_preds_int)
    print(f"\n>>> Final System QWK on Demo Validation Set: {final_qwk:.4f}")

    # Final Validation
    assert len(final_preds_int) == len(df_val)
    assert np.all((final_preds_int >= 1) & (final_preds_int <= 6))
    print(">>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    run_demonstration()
