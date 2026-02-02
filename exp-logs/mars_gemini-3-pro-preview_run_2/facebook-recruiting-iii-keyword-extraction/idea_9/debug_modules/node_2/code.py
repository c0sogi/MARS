import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

# Import library components
from library.config import Config
from library.utils import set_seed, get_dynamic_threshold
from library.data_factory import DataFactory, TextPreprocessor
from library.datasets import WideDataset, DeepDataset
from library.models import WideLinear, DeepTransformer
from library.training import Trainer, FocalLoss, predict_logits


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Demo
    # -------------------------------------------------------------------------
    print(">>> Configuring demo parameters...")
    # Override Config values to ensure quick execution
    Config.DEEP_SUBSET_SIZE = 200  # Train on only 200 samples for Deep model
    Config.TFIDF_MAX_FEATURES = 1000  # Reduce vocabulary size for Wide model
    Config.WIDE_EPOCHS = 1  # Train for 1 epoch
    Config.DEEP_EPOCHS = 1  # Train for 1 epoch
    Config.WIDE_BATCH_SIZE = 128  # Smaller batch size for demo
    Config.DEEP_BATCH_SIZE = 8  # Smaller batch size for demo
    Config.WORKING_DIR = "./working/demo_run"  # Isolate demo outputs

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print(">>> Initializing DataFactory...")
    factory = DataFactory()

    # --- Wide Data Preparation ---
    print(">>> Preparing Wide features (Train/Val)...")
    # load_cached_data=False forces re-computation with new TFIDF_MAX_FEATURES
    X_wide_train = factory.get_wide_features("train", load_cached_data=False)
    y_train = factory.get_targets("train", load_cached_data=False)

    X_wide_val = factory.get_wide_features("val", load_cached_data=False)
    y_val = factory.get_targets("val", load_cached_data=False)

    # Assertions to verify logic
    assert (
        X_wide_train.shape[0] == y_train.shape[0]
    ), "Train features and targets count mismatch"
    assert (
        X_wide_train.shape[1] == Config.TFIDF_MAX_FEATURES
    ), f"Expected {Config.TFIDF_MAX_FEATURES} features, got {X_wide_train.shape[1]}"

    # --- Deep Data Preparation ---
    print(">>> Preparing Deep data (Train/Val)...")
    df_deep_train = factory.get_deep_data("train", load_cached_data=False)
    df_deep_val = factory.get_deep_data("val", load_cached_data=False)

    assert (
        len(df_deep_train) == Config.DEEP_SUBSET_SIZE
    ), "Deep train subset size mismatch"

    # For the demo, we also subsample validation data for the Deep model to save time
    val_subset_size = 200
    df_deep_val_sub = df_deep_val.iloc[:val_subset_size].copy()

    # We also need to slice the Wide validation data and Targets to match this subset
    X_wide_val_sub = X_wide_val[:val_subset_size]
    y_val_sub = y_val[:val_subset_size]

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------

    # --- Train Wide Model ---
    print(">>> Training Wide Model...")
    wide_train_ds = WideDataset(X_wide_train, y_train)
    # Use the subset for validation during training to speed up the loop
    wide_val_ds = WideDataset(X_wide_val_sub, y_val_sub)

    wide_train_loader = DataLoader(
        wide_train_ds, batch_size=Config.WIDE_BATCH_SIZE, shuffle=True
    )
    wide_val_loader = DataLoader(
        wide_val_ds, batch_size=Config.WIDE_BATCH_SIZE, shuffle=False
    )

    wide_model = WideLinear(
        input_dim=X_wide_train.shape[1], output_dim=y_train.shape[1]
    ).to(Config.DEVICE)
    wide_optimizer = AdamW(wide_model.parameters(), lr=Config.WIDE_LR)
    criterion = FocalLoss()

    wide_trainer = Trainer(
        wide_model,
        wide_optimizer,
        criterion,
        Config.DEVICE,
        save_path=os.path.join(Config.WORKING_DIR, "wide_model.pth"),
    )

    wide_trainer.fit(wide_train_loader, wide_val_loader, epochs=Config.WIDE_EPOCHS)

    # --- Train Deep Model ---
    print(">>> Training Deep Model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tag_encoder = factory.get_tag_encoder()

    deep_train_ds = DeepDataset(
        df_deep_train["text"], df_deep_train["tags"], tokenizer, tag_encoder
    )
    deep_val_ds = DeepDataset(
        df_deep_val_sub["text"], df_deep_val_sub["tags"], tokenizer, tag_encoder
    )

    deep_train_loader = DataLoader(
        deep_train_ds, batch_size=Config.DEEP_BATCH_SIZE, shuffle=True
    )
    deep_val_loader = DataLoader(
        deep_val_ds, batch_size=Config.DEEP_BATCH_SIZE, shuffle=False
    )

    deep_model = DeepTransformer(num_classes=len(tag_encoder.classes_)).to(
        Config.DEVICE
    )
    deep_optimizer = AdamW(deep_model.parameters(), lr=Config.DEEP_LR)

    deep_trainer = Trainer(
        deep_model,
        deep_optimizer,
        criterion,
        Config.DEVICE,
        save_path=os.path.join(Config.WORKING_DIR, "deep_model.pth"),
    )

    deep_trainer.fit(deep_train_loader, deep_val_loader, epochs=Config.DEEP_EPOCHS)

    # -------------------------------------------------------------------------
    # 4. Validation & Threshold Optimization
    # -------------------------------------------------------------------------
    print(">>> Optimizing Thresholds...")

    # Get logits
    wide_logits = predict_logits(wide_model, wide_val_loader, Config.DEVICE)
    deep_logits = predict_logits(deep_model, deep_val_loader, Config.DEVICE)

    # Simple Ensemble: Average Logits
    ensemble_logits = (wide_logits + deep_logits) / 2.0
    ensemble_probs = torch.sigmoid(torch.from_numpy(ensemble_logits)).numpy()

    # Find best threshold
    # Note: y_val_sub is sparse, get_dynamic_threshold expects dense or array-like
    y_true_dense = y_val_sub.toarray()
    best_thresh, best_f1 = get_dynamic_threshold(y_true_dense, ensemble_probs)

    print(f"Optimal Threshold: {best_thresh:.4f} (Val F1: {best_f1:.4f})")

    # -------------------------------------------------------------------------
    # 5. Inference on Test Subset
    # -------------------------------------------------------------------------
    print(">>> Running Inference on Test Subset...")

    # Load a small subset of test data
    df_test = factory._load_raw_df("test").iloc[:50].copy()

    # Preprocess text
    clean_text_test = TextPreprocessor.preprocess_dataframe(df_test)

    # Prepare Wide Features for Test
    # Use the fitted vectorizer from factory
    X_test_wide = factory.tfidf_featurizer.transform(clean_text_test)

    # Prepare Datasets
    test_wide_ds = WideDataset(X_test_wide)  # No targets
    test_deep_ds = DeepDataset(clean_text_test, None, tokenizer, tag_encoder)

    test_wide_loader = DataLoader(
        test_wide_ds, batch_size=Config.WIDE_BATCH_SIZE, shuffle=False
    )
    test_deep_loader = DataLoader(
        test_deep_ds, batch_size=Config.DEEP_BATCH_SIZE, shuffle=False
    )

    # Predict
    test_wide_logits = predict_logits(wide_model, test_wide_loader, Config.DEVICE)
    test_deep_logits = predict_logits(deep_model, test_deep_loader, Config.DEVICE)

    # Ensemble
    test_ensemble_logits = (test_wide_logits + test_deep_logits) / 2.0
    test_probs = torch.sigmoid(torch.from_numpy(test_ensemble_logits)).numpy()

    # Apply Threshold
    test_preds_bin = (test_probs >= best_thresh).astype(int)

    # Decode Tags
    pred_tags_list = tag_encoder.inverse_transform(test_preds_bin)

    # Create Submission DataFrame
    submission = pd.DataFrame({"Id": df_test["Id"], "Tags": pred_tags_list})

    print("\nSample Predictions:")
    print(submission.head())

    # Save
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")


if __name__ == "__main__":
    run_demo()
