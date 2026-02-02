import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, compute_score, jaccard
from library.data import load_and_process_data, QADataset
from library.model import QAModel
from library.engine import train_loop, generate_submission


def create_subset_data(source_path, dest_path, n=20):
    """Helper to create small data subsets for speed."""
    if os.path.exists(source_path):
        df = pd.read_csv(source_path)
        # Sample n rows, or all if len < n
        subset = df.head(min(len(df), n)).copy()
        subset.to_csv(dest_path, index=False)
        print(f"Created subset: {dest_path} with {len(subset)} rows.")
        return len(subset)
    return 0


if __name__ == "__main__":
    # 1. Setup and Reproducibility
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Prepare Data Subsets (Optimization for Speed)
    # We use the metadata files as source since they are guaranteed to exist and be split correctly.
    # We save them to ./working so we don't modify input/metadata.
    working_dir = "./working/demo_run"
    os.makedirs(working_dir, exist_ok=True)

    demo_train_path = os.path.join(working_dir, "train_subset.csv")
    demo_val_path = os.path.join(working_dir, "val_subset.csv")
    demo_test_path = os.path.join(working_dir, "test_subset.csv")

    # Create subsets
    n_train = create_subset_data("./metadata/train.csv", demo_train_path, n=20)
    n_val = create_subset_data("./metadata/val.csv", demo_val_path, n=10)
    n_test = create_subset_data("./metadata/test.csv", demo_test_path, n=10)

    # 3. Configure the Run
    # IMPORTANT: The library.model.post_process_predictions function reads
    # Config.test_data_path directly (static access). We must patch the class attribute.
    Config.test_data_path = demo_test_path

    # Instantiate and override instance config
    config = Config()
    config.working_dir = working_dir
    config.train_data_path = demo_train_path
    config.val_data_path = demo_val_path
    config.test_data_path = demo_test_path  # Instance variable update as well

    # Override hyperparameters for speed
    config.epochs = 1
    config.batch_size = 4
    config.n_folds = 2  # Minimum for GroupKFold

    # Update cache paths to avoid conflicts with other runs
    config.train_cache_path = os.path.join(working_dir, "train.parquet")
    config.val_cache_path = os.path.join(working_dir, "val.parquet")
    config.test_cache_path = os.path.join(working_dir, "test.parquet")
    config.submission_file = os.path.join(working_dir, "submission.csv")

    print("\n=== Configuration ===")
    print(f"Epochs: {config.epochs}")
    print(f"Folds: {config.n_folds}")
    print(f"Model: {config.model_checkpoint}")

    # 4. Data Loading and Processing
    print("\n=== Processing Data ===")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    # Force re-processing by ignoring cache (or since we changed paths, it won't find them)
    train_features, val_features, test_features = load_and_process_data(
        config, tokenizer, load_cached_data=False
    )

    # Verify Data Processing
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    assert not train_features.empty, "Train features should not be empty"
    assert "input_ids" in train_features.columns
    assert "start_positions" in train_features.columns
    assert "fold" in train_features.columns

    # 5. Model Initialization
    print("\n=== Initializing Model ===")
    model = QAModel(config)
    model.to(device)

    # Verify Model Output Shape
    # Create a dummy batch from train features
    dummy_dataset = QADataset(train_features.head(2), is_test=False)
    dummy_loader = DataLoader(dummy_dataset, batch_size=2)
    dummy_batch = next(iter(dummy_loader))

    with torch.no_grad():
        start_logits, end_logits = model(
            dummy_batch["input_ids"].to(device),
            dummy_batch["attention_mask"].to(device),
        )

    print(f"Logits shape: {start_logits.shape}")
    assert start_logits.shape == (2, config.max_length), "Incorrect start logits shape"
    assert end_logits.shape == (2, config.max_length), "Incorrect end logits shape"

    # 6. Training Loop (Single Fold Demo)
    print("\n=== Starting Training (Fold 0) ===")
    # We will simulate the loop for Fold 0 manually using library components
    fold = 0
    fold_train_data = train_features[train_features["fold"] != fold].reset_index(
        drop=True
    )
    fold_val_data = train_features[train_features["fold"] == fold].reset_index(
        drop=True
    )

    # If fold split resulted in empty set (due to very small data), use full set for demo
    if fold_train_data.empty:
        fold_train_data = train_features
    if fold_val_data.empty:
        fold_val_data = train_features

    train_ds = QADataset(fold_train_data, is_test=False)
    val_ds = QADataset(fold_val_data, is_test=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Train using the engine's train_loop
    trained_model = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        patience=1,
    )

    # 7. Inference and Submission
    print("\n=== Generating Submission ===")
    test_ds = QADataset(test_features, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    generate_submission(
        model=trained_model,
        test_loader=test_loader,
        test_features=test_features,
        config=config,
        output_file=config.submission_file,
    )

    # 8. Validation of Results
    print("\n=== Validating Results ===")

    # Check if file exists
    assert os.path.exists(config.submission_file), "Submission file was not created."

    # Load submission
    sub_df = pd.read_csv(config.submission_file)
    print("Submission Head:")
    print(sub_df.head())

    # Check format
    assert "id" in sub_df.columns and "PredictionString" in sub_df.columns
    assert (
        len(sub_df) == 112
    ), "Submission should contain all rows from sample_submission (112 rows)"

    # Check logic of metric functions
    print("\n=== Testing Metric Functions ===")
    s1 = "India is a country"
    s2 = "India country"
    score = jaccard(s1, s2)
    print(f"Jaccard('{s1}', '{s2}') = {score:.4f}")

    # Expected: Intersection {india, country} (2) / Union {india, is, a, country} (4) = 0.5
    assert abs(score - 0.5) < 1e-6, "Jaccard calculation is incorrect"

    avg_score = compute_score([s1], [s2])
    assert abs(avg_score - 0.5) < 1e-6, "Compute score calculation is incorrect"

    print("\nAll demonstrations and validations passed successfully.")
