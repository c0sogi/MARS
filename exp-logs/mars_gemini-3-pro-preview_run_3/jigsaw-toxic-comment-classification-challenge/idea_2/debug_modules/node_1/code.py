import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from scipy import sparse

# Import provided library modules
from library import utils, data_processing, model_definitions, training_engine


def run_demo():
    # 1. Setup and Configuration
    print("=== Step 1: Setup and Data Subsetting ===")
    utils.seed_everything(42)

    # Load the full metadata-based dataframes
    print("Loading original data structure...")
    train_full, val_full, test_full = data_processing.load_data_from_metadata()

    # Create small subsets for speed (200 train, 50 val, 50 test)
    # This ensures the demo finishes in < 5 minutes on GPU

    # Ensure training subset contains at least 2 positive samples for each class
    # to avoid ValueError in LogisticRegression (needs >= 2 classes).
    label_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    train_indices = set()

    for col in label_cols:
        pos_rows = train_full[train_full[col] == 1].index
        train_indices.update(pos_rows[:2].tolist())

    for idx in train_full.index:
        if len(train_indices) >= 200:
            break
        train_indices.add(idx)

    train_subset = (
        train_full.loc[list(train_indices)]
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    val_subset = val_full.iloc[:50].reset_index(drop=True)
    test_subset = test_full.iloc[:50].reset_index(drop=True)

    print(
        f"Subset shapes - Train: {train_subset.shape}, Val: {val_subset.shape}, Test: {test_subset.shape}"
    )

    # 2. Demonstrate NBSVM Component
    print("\n=== Step 2: Demonstrating NBSVM Component ===")

    # Generate features for the subset
    # We set load_cached_data=False to force generation on this new subset
    # Note: The library caches to ./working/idea_2.
    print("Generating TF-IDF features for subset...")
    train_feats, val_feats, test_feats = data_processing.get_tfidf_features(
        train_subset, val_subset, test_subset, load_cached_data=False
    )

    # Verify feature shapes
    assert train_feats.shape[0] == 200
    assert val_feats.shape[0] == 50
    assert test_feats.shape[0] == 50
    print("TF-IDF features generated successfully.")

    # Instantiate and fit NBSVM
    print("Fitting NBSVM on subset...")
    nbsvm = model_definitions.NBSVM(C=1.0, dual=True, n_jobs=1)

    # Extract targets
    label_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    y_train = train_subset[label_cols].values

    nbsvm.fit(train_feats, y_train)

    # Predict
    val_probs = nbsvm.predict_proba(val_feats)

    # Verify predictions
    assert val_probs.shape == (50, 6), f"Expected shape (50, 6), got {val_probs.shape}"
    assert np.all(
        (val_probs >= 0) & (val_probs <= 1)
    ), "Probabilities must be in [0, 1]"
    print(f"NBSVM predictions verified. Sample: {val_probs[0]}")

    # 3. Demonstrate RoBERTa Component
    print("\n=== Step 3: Demonstrating RoBERTa Component ===")

    # Initialize Tokenizer
    from transformers import AutoTokenizer

    model_name = "roberta-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError:
        print(
            "Warning: Could not download tokenizer (no internet?). Using basic whitespace tokenization for demo if needed."
        )
        # In a real environment with the provided packages, this should work.
        # If not, we would need a fallback, but we assume requirements are met.
        raise

    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = data_processing.make_dataloaders(
        train_subset, val_subset, test_subset, tokenizer, batch_size=4, max_len=64
    )

    # Check one batch
    batch = next(iter(train_loader))
    ids = batch["ids"]
    mask = batch["mask"]
    targets = batch["targets"]

    print(
        f"Batch shapes - IDs: {ids.shape}, Mask: {mask.shape}, Targets: {targets.shape}"
    )
    assert ids.shape == (4, 64)
    assert targets.shape == (4, 6)

    # Initialize Model
    print("Initializing ToxicRoBERTa model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_definitions.ToxicRoBERTa(model_name=model_name, num_classes=6)
    model.to(device)
    model.eval()

    # Forward pass check
    with torch.no_grad():
        ids = ids.to(device)
        mask = mask.to(device)
        logits = model(ids, mask)

    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (4, 6)
    print("RoBERTa forward pass successful.")

    # 4. Demonstrate Full Pipeline (Training Engine)
    print("\n=== Step 4: Running Full Pipeline (Simulated) ===")

    # We use monkey-patching to inject our subsets into the training_engine.
    # This allows us to use the high-level 'run_full_training' function
    # without it trying to process the entire 150k+ dataset.

    original_loader = training_engine.load_data_from_metadata

    def mock_loader():
        print(">> Using mocked data loader with subsets...")
        return train_subset, val_subset, test_subset

    # Apply patch
    training_engine.load_data_from_metadata = mock_loader

    try:
        # Run the full training loop
        # We use minimal epochs and batch size for speed
        print("Starting execution of training_engine.run_full_training...")

        # Note: We set load_cached_features=True because we already generated and cached
        # the subset features in Step 2 (get_tfidf_features caches to ./working/idea_2).
        final_preds = training_engine.run_full_training(
            load_cached_features=True,
            roberta_epochs=1,
            roberta_batch_size=8,
            ensemble_alpha=0.5,
            seed=42,
        )

        # Verify output
        assert final_preds.shape == (50, 6)
        print("Pipeline execution completed successfully.")

        # Verify submission file creation
        submission_path = "./submission/submission.csv"
        if os.path.exists(submission_path):
            print(f"Submission file found at {submission_path}")
            sub_df = pd.read_csv(submission_path)
            assert sub_df.shape == (50, 7)  # id + 6 labels
            print("Submission file content verified.")
        else:
            raise FileNotFoundError("Submission file was not created.")

    finally:
        # Restore original function (good practice)
        training_engine.load_data_from_metadata = original_loader


if __name__ == "__main__":
    run_demo()
