import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# Import from the provided library files
from library.utils import set_seed, clean_text, calculate_auc, load_data
from library.data_factory import get_tfidf_features, InsultDataset
from library.modeling import NBSVM, RoBERTaClassifier
from library.execution import optimize_ensemble


def main():
    # 1. Setup
    print("--- 1. Setup & Configuration ---")
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading & Preprocessing Demo
    print("\n--- 2. Data Loading & Utils ---")

    # Test clean_text
    raw_text = "Hello\\nWorld"
    cleaned = clean_text(raw_text)
    print(f"Raw: {raw_text} -> Cleaned: {cleaned}")
    assert cleaned == "Hello\nWorld", "clean_text failed to decode newline"

    # Load Data (using metadata)
    # Note: load_data caches to the specified directory. We use our demo dir.
    print("Loading data...")
    train_df = load_data("train", load_cached_data=False, cache_dir=DEMO_DIR)
    val_df = load_data("val", load_cached_data=False, cache_dir=DEMO_DIR)
    test_df = load_data("test", load_cached_data=False, cache_dir=DEMO_DIR)

    # Create tiny subsets for speed
    subset_size = 50
    train_sub = train_df.head(subset_size).copy()
    val_sub = val_df.head(subset_size).copy()
    test_sub = test_df.head(subset_size).copy()

    print(
        f"Subset shapes: Train {train_sub.shape}, Val {val_sub.shape}, Test {test_sub.shape}"
    )
    assert len(train_sub) == subset_size

    # 3. Statistical Stream (TF-IDF + NBSVM)
    print("\n--- 3. Statistical Stream (NBSVM) ---")

    # Generate Features
    # We use a separate cache dir for features to avoid conflicts
    feat_cache_dir = os.path.join(DEMO_DIR, "tfidf_feats")
    X_train, y_train, X_val, y_val, X_test = get_tfidf_features(
        train_sub, val_sub, test_sub, load_cached_data=False, cache_dir=feat_cache_dir
    )

    print(f"TF-IDF Train Shape: {X_train.shape}")
    # Check that we have features (cols > 0) and samples match subset_size
    assert X_train.shape[0] == subset_size
    assert X_train.shape[1] > 0
    assert y_train.shape[0] == subset_size

    # Initialize and Train NBSVM
    print("Training NBSVM...")
    nbsvm = NBSVM(C=1.0, dual=False, random_state=42)  # dual=False for lbfgs/primal
    nbsvm.fit(X_train, y_train)

    # Predict
    nb_val_probs = nbsvm.predict_proba(X_val)[:, 1]

    print(f"NBSVM Preds (First 5): {nb_val_probs[:5]}")
    assert np.all(
        (nb_val_probs >= 0) & (nb_val_probs <= 1)
    ), "NBSVM probabilities out of range"

    # Calculate AUC
    nb_auc = calculate_auc(y_val, nb_val_probs)
    print(f"NBSVM Subset AUC: {nb_auc:.4f}")

    # 4. Neural Stream (RoBERTa)
    print("\n--- 4. Neural Stream (RoBERTa) ---")

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Dataset
    train_dataset = InsultDataset(train_sub, tokenizer, max_len=64)
    val_dataset = InsultDataset(val_sub, tokenizer, max_len=64)

    # Check a single item
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "target" in sample_item
    assert sample_item["input_ids"].dim() == 1

    # DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = RoBERTaClassifier(model_name="roberta-base", dropout=0.1)
    model.to(device)

    # Train (1 Epoch for demo)
    print("Training RoBERTa (1 Epoch)...")
    best_auc = model.train_model(
        train_loader, val_loader, device, epochs=1, lr=1e-5, patience=1
    )

    # Predict
    neural_val_probs = model.predict(val_loader, device)

    print(f"Neural Preds (First 5): {neural_val_probs[:5]}")
    assert len(neural_val_probs) == subset_size
    assert np.all(
        (neural_val_probs >= 0) & (neural_val_probs <= 1)
    ), "Neural probabilities out of range"

    # 5. Ensemble Optimization
    print("\n--- 5. Ensemble Optimization ---")

    # We use the validation labels and the predictions from both models
    # Note: In a real scenario, we'd want more data for a stable weight,
    # but this demonstrates the function call.
    best_w, ensemble_auc = optimize_ensemble(y_val, nb_val_probs, neural_val_probs)

    print(f"Optimal Weight (NBSVM): {best_w}")
    print(f"Ensemble AUC: {ensemble_auc:.4f}")

    assert 0.0 <= best_w <= 1.0, "Optimal weight out of range [0, 1]"

    # 6. Cleanup
    print("\n--- 6. Cleanup ---")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
        print(f"Removed {DEMO_DIR}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
