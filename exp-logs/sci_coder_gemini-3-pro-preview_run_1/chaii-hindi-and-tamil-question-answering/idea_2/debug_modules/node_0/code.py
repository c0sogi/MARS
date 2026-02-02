import os
import sys
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.model_handler import get_tokenizer, get_model
from library.data_loader import get_processed_data, QADataset
from library.trainer import train_fn, eval_fn, predict_fn
from library.post_processor import save_submission


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("Configuring environment...")

    # Set up a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths
    Config.working_dir = demo_dir
    Config.train_cache_path = os.path.join(demo_dir, "train_processed.parquet")
    Config.val_cache_path = os.path.join(demo_dir, "val_processed.parquet")
    Config.test_cache_path = os.path.join(demo_dir, "test_processed.parquet")
    Config.model_output_dir = os.path.join(demo_dir, "best_model")
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for Speed
    Config.debug = True
    Config.debug_subset_size = 20  # Use only 20 samples
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.eval_batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small data

    # Set seed
    set_seed(Config.seed)

    # -------------------------------------------------------------------------
    # 2. Data Processing & Loading
    # -------------------------------------------------------------------------
    print("\nProcessing data...")
    tokenizer = get_tokenizer()

    # Generate features. We set load_cached_data=False to force processing the debug subset.
    train_features, val_features, test_features = get_processed_data(
        tokenizer, load_cached_data=False
    )

    # Validation: Check if features are generated
    assert not train_features.empty, "Train features DataFrame is empty."
    assert not val_features.empty, "Val features DataFrame is empty."
    assert not test_features.empty, "Test features DataFrame is empty."

    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")

    # Create Datasets
    train_dataset = QADataset(train_features, mode="train")
    val_dataset = QADataset(val_features, mode="val")
    test_dataset = QADataset(test_features, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.eval_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.eval_batch_size, shuffle=False
    )

    # Validation: Check a single batch
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch, "Batch missing input_ids"
    assert "start_positions" in sample_batch, "Batch missing start_positions"
    print("DataLoader verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing model...")
    device = Config.device
    model = get_model()
    model.to(device)

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    print(f"Model loaded on {device}.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\nRunning Training Step...")
    # Run one epoch (which is very short due to debug subset)
    avg_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch=0)

    # Validation: Loss should be a valid float
    assert isinstance(avg_loss, float), "Training loss is not a float."
    assert not pd.isna(avg_loss), "Training loss is NaN."
    print(f"Training step complete. Loss: {avg_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Evaluation Demonstration
    # -------------------------------------------------------------------------
    print("\nRunning Evaluation Step...")

    # Load raw validation data (needed for text extraction in eval_fn)
    # Since we used debug=True in get_processed_data, we must load the same subset
    raw_val = pd.read_csv(Config.val_path).head(Config.debug_subset_size)

    # Run evaluation
    val_score = eval_fn(model, val_loader, raw_val, val_features, device)

    # Validation: Score should be between 0 and 1
    assert 0.0 <= val_score <= 1.0, f"Validation score {val_score} out of range."
    print(f"Evaluation step complete. Jaccard Score: {val_score:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\nRunning Inference Step...")

    # Load raw test data
    raw_test = pd.read_csv(Config.test_path).head(Config.debug_subset_size)

    # Run prediction
    predictions = predict_fn(model, test_loader, raw_test, test_features, device)

    # Validation: Predictions should be a dictionary matching raw_test IDs
    assert isinstance(predictions, dict), "Predictions should be a dictionary."
    assert len(predictions) > 0, "No predictions generated."

    # Check if a specific ID from raw_test is in predictions
    sample_id = raw_test.iloc[0]["id"]
    assert sample_id in predictions, f"ID {sample_id} missing from predictions."
    print(f"Inference step complete. Generated {len(predictions)} predictions.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\nGenerating Submission File...")
    save_submission(predictions, Config.submission_path)

    # Validation: Check file existence
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    # Check file content format
    sub_df = pd.read_csv(Config.submission_path)
    assert list(sub_df.columns) == [
        "id",
        "PredictionString",
    ], "Submission columns incorrect."
    assert len(sub_df) == len(predictions), "Submission row count mismatch."

    print(f"Submission saved to {Config.submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
