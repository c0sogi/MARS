import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import provided library modules
from library.utils import seed_everything
from library.cpc_utils import get_cpc_text
from library.feature_engineering import StructuralFeatureEngineer
from library.dataset import get_datasets
from library.model import DebertaV3FeatureFused
from library.engine import train_one_epoch, validate_one_epoch, predict


def main():
    # 1. Configuration & Setup
    print("Setting up environment...")
    SEED = 42
    seed_everything(SEED)

    # Suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use a smaller model for the demo to ensure speed
    MODEL_NAME = "microsoft/deberta-v3-small"
    BATCH_SIZE = 8
    MAX_LENGTH = 64  # Reduced length for speed
    LR = 2e-5
    EPOCHS = 1

    # 2. Verify Utility Functions
    print("Verifying utilities...")
    # Check CPC mapping
    code = "A47"
    text = get_cpc_text(code)
    assert isinstance(text, str) and len(text) > 0, "CPC text retrieval failed"
    print(f"CPC Code '{code}' mapped to: {text[:50]}...")

    # Check Feature Engineering
    print("Verifying feature engineering...")
    dummy_df = pd.DataFrame(
        {
            "anchor": ["test phrase", "another one"],
            "target": ["test phrase", "completely different"],
        }
    )
    engineer = StructuralFeatureEngineer(cache_dir="./working/debug_feats")
    feats = engineer.compute_features(dummy_df)

    assert feats.shape == (2, 3), f"Expected shape (2, 3), got {feats.shape}"
    assert "norm_levenshtein" in feats.columns
    assert (
        feats.iloc[0]["norm_levenshtein"] == 0.0
    ), "Identical strings should have 0 distance"
    # Normalized Levenshtein for identical strings is 0.0
    # Jaccard for identical strings is 1.0
    assert (
        feats.iloc[0]["jaccard_sim"] == 1.0
    ), "Identical strings should have 1.0 Jaccard similarity"

    # 3. Data Loading
    print("Loading datasets (Debug Mode)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # debug=True loads a tiny subset (100 train, 50 val, 50 test)
    train_ds, val_ds, test_ds = get_datasets(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        load_cached_data=False,  # Force re-compute for demo purposes
        debug=True,
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "structural_features" in sample_batch
    assert "label" in sample_batch
    assert sample_batch["structural_features"].shape[1] == 3
    print("Batch structure verified.")

    # 4. Model Initialization
    print("Initializing model...")
    model = DebertaV3FeatureFused(
        model_name=MODEL_NAME,
        num_classes=5,  # 0.0, 0.25, 0.5, 0.75, 1.0
        num_structural_features=3,
        pretrained=True,
    )
    model.to(device)

    # 5. Training Loop
    print("Starting training...")
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )
        assert not np.isnan(train_loss), "Training loss is NaN"

        # Validate
        val_loss, val_pearson = validate_one_epoch(model, val_loader, device)
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert -1.0 <= val_pearson <= 1.0, f"Pearson score {val_pearson} out of range"

        print(
            f"Epoch {epoch} Summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Pearson={val_pearson:.4f}"
        )

    # 6. Prediction
    print("\nGenerating predictions on test set...")
    test_preds = predict(model, test_loader, device)

    assert len(test_preds) == len(
        test_ds
    ), "Mismatch between predictions and test set size"

    # 7. Create Submission
    print("Creating submission file...")
    # We need the IDs from the test dataframe.
    # Since dataset wraps the dataframe, we can access it directly or reload the metadata.
    # The dataset object stores the dataframe in .df
    test_ids = test_ds.df["id"].values

    submission_df = pd.DataFrame({"id": test_ids, "score": test_preds})

    # Ensure working directory exists
    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
