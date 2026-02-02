import os
import sys
import torch
import pandas as pd
import transformers
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed, jaccard, compute_average_jaccard
from library.model_factory import get_model, get_tokenizer
from library.data_factory import get_train_dataset, get_val_dataset, get_test_dataset
from library.trainer import TrainRunner
from library.predictor import InferenceEngine, generate_submission

# Suppress verbose output from transformers
transformers.logging.set_verbosity_error()


def main():
    print("=== Starting Demonstration of QA Pipeline ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment...")
    set_seed(42)

    # Override Config for a quick demo run
    Config.debug = True
    Config.debug_sample_size = 20  # Use very small subset
    Config.epochs = 1
    Config.n_folds = 1  # Only run one fold
    Config.train_batch_size = 4
    Config.eval_batch_size = 4
    Config.output_dir = "./working/demo_run"

    # Update paths to use the demo output directory
    Config.cached_train_features_path = os.path.join(
        Config.output_dir, "cached_train_features.parquet"
    )
    Config.cached_test_features_path = os.path.join(
        Config.output_dir, "cached_test_features.parquet"
    )
    Config.score_tracking_path = os.path.join(Config.output_dir, "best_cv_score.txt")

    os.makedirs(Config.output_dir, exist_ok=True)
    print(f"Debug mode: {Config.debug}")
    print(f"Output directory: {Config.output_dir}")

    # 2. Metric Verification
    print("\n[2] Verifying Metric Functions...")
    s1 = "This is a test answer"
    s2 = "This is a test"
    score = jaccard(s1, s2)
    # Intersection: {this, is, a, test} (4), Union: {this, is, a, test, answer} (5) -> 4/5 = 0.8
    print(f"Jaccard('{s1}', '{s2}') = {score}")
    assert abs(score - 0.8) < 1e-6, "Jaccard calculation is incorrect"

    avg_score = compute_average_jaccard([s1], [s2])
    assert abs(avg_score - 0.8) < 1e-6, "Average Jaccard calculation is incorrect"
    print("Metric verification passed.")

    # 3. Data Preparation
    print("\n[3] Loading and Processing Data...")
    tokenizer = get_tokenizer()

    # Load Datasets (this will trigger caching logic in data_factory)
    train_dataset = get_train_dataset(tokenizer, load_cached_data=False)
    val_dataset, raw_val_df, val_features_df = get_val_dataset(
        tokenizer, load_cached_data=False
    )
    test_dataset, raw_test_df, test_features_df = get_test_dataset(
        tokenizer, load_cached_data=False
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    # Note: TrainRunner expects a custom collate_fn if batches are variable length,
    # but since data_factory pads to max_length, default collate works for tensors.
    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.eval_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.eval_batch_size, shuffle=False
    )

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch, "Batch missing input_ids"
    assert "start_positions" in sample_batch, "Training batch missing labels"
    print("DataLoaders created successfully.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = get_model()
    model.to(Config.device)

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    print(f"Model loaded: {Config.model_checkpoint}")
    print(f"Device: {Config.device}")

    # 5. Training Loop Execution
    print("\n[5] Running Training Loop (1 Epoch)...")
    runner = TrainRunner(model, tokenizer, optimizer, scheduler, Config.device)

    # Run training for Fold 0
    # This will train, validate, and save 'best_model_fold_0.pth' to output_dir
    best_score = runner.run(
        train_loader, val_loader, val_features_df, raw_val_df, fold_idx=0
    )

    print(f"Training complete. Best Validation Score: {best_score:.4f}")

    expected_model_path = os.path.join(Config.output_dir, "best_model_fold_0.pth")
    assert os.path.exists(expected_model_path), "Model checkpoint was not saved."
    print(f"Checkpoint verified at: {expected_model_path}")

    # 6. Inference Engine Usage
    print("\n[6] Running Inference Engine...")
    predictor = InferenceEngine(device=Config.device)

    # Load the model we just trained (Fold 0)
    # We set num_folds=1 so it only looks for fold_0
    predictor.load_ensemble(num_folds=1)

    # Generate predictions on Test set
    predictions = predictor.predict(test_loader, test_features_df, raw_test_df)

    print(f"Generated predictions for {len(predictions)} examples.")
    assert len(predictions) > 0, "No predictions generated."

    # Check format of predictions
    sample_id = list(predictions.keys())[0]
    sample_pred = predictions[sample_id]
    print(f"Sample Prediction -> ID: {sample_id}, Answer: '{sample_pred}'")
    assert isinstance(sample_pred, str), "Prediction is not a string"

    # 7. Submission Generation
    print("\n[7] Generating Submission File...")
    submission_path = os.path.join(Config.output_dir, "submission.csv")
    generate_submission(predictions, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file not created."

    # Validate submission file content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission file shape: {sub_df.shape}")
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Invalid submission columns"
    assert len(sub_df) == len(predictions), "Submission row count mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
