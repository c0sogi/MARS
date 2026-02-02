import os
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import get_linear_schedule_with_warmup

# Import from provided library
from library.config import Config
from library.utils import seed_everything, compute_score, get_cpc_mapping
from library.dataset import get_dataloaders
from library.model import CustomDeberta
from library.engine import train_model, generate_submission, get_expected_scores


def run_demo():
    print("--- Starting Demo of Phrase Similarity Task Pipeline ---")

    # =========================================================================
    # 1. Configure for Speed and Demo
    # =========================================================================
    print("1. Configuring environment for demo...")

    # Override Config attributes to run a quick functional test
    Config.debug = True
    Config.debug_sample_size = 50  # Small subset for speed
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8

    # Use a smaller model for the demo to ensure it fits in memory/time constraints easily
    # while maintaining architectural compatibility (DeBERTa V3)
    Config.model_name = "microsoft/deberta-v3-xsmall"

    # Use a separate working directory for the demo
    Config.working_dir = "./working/demo_run"
    Config.setup()  # Ensure directory exists

    # Suppress verbose logs
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # =========================================================================
    # 2. Test Utils
    # =========================================================================
    print("2. Testing Utility functions...")
    seed_everything(Config.seed)

    # Test compute_score with dummy data
    y_true = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_pred = [0.0, 0.2, 0.6, 0.8, 0.9]
    score = compute_score(y_true, y_pred)
    print(f"   Dummy Pearson Score: {score:.4f}")
    assert isinstance(score, float), "Score should be a float"

    # Test CPC mapping
    mapping = get_cpc_mapping()
    assert "A01" in mapping, "CPC Mapping should contain 'A01'"
    print("   CPC Mapping loaded successfully.")

    # =========================================================================
    # 3. Test Dataset Loading
    # =========================================================================
    print("3. Testing Data Loading...")
    # This will trigger processing and caching of the debug subset
    # We set load_cached_data=False to force processing logic verification
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"   Train Batches: {len(train_loader)}")
    print(f"   Val Batches: {len(val_loader)}")
    print(f"   Test Batches: {len(test_loader)}")

    # Verify a single batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print(f"   Input Shape: {input_ids.shape}")
    print(f"   Labels Shape: {labels.shape}")

    assert input_ids.shape[0] <= Config.train_batch_size, "Batch size mismatch"
    assert (
        labels.min() >= 0 and labels.max() < Config.num_classes
    ), "Labels out of range"

    # =========================================================================
    # 4. Test Model Initialization
    # =========================================================================
    print("4. Testing Model Initialization...")
    device = Config.device

    # Initialize model with the overridden model_name (xsmall)
    model = CustomDeberta(model_name=Config.model_name, num_classes=Config.num_classes)
    model.to(device)

    # Verify Forward Pass
    print("   Running forward pass check...")
    with torch.no_grad():
        # Move dummy batch to device
        ids_d = input_ids.to(device)
        mask_d = attention_mask.to(device)
        logits = model(ids_d, mask_d)

    print(f"   Logits Shape: {logits.shape}")
    assert logits.shape[1] == Config.num_classes, "Logits output dimension mismatch"

    # Test Expected Score Calculation
    exp_scores = get_expected_scores(logits.cpu())
    assert len(exp_scores) == logits.shape[0]
    assert (exp_scores >= 0).all() and (
        exp_scores <= 1
    ).all(), "Expected scores out of range [0,1]"

    # =========================================================================
    # 5. Test Training Loop (Engine)
    # =========================================================================
    print("5. Testing Training Loop...")

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * Config.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # Run Training
    # train_model returns the model with best weights loaded
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.epochs,
        patience=1,
        fold=0,
    )
    print("   Training cycle completed.")

    # =========================================================================
    # 6. Test Submission Generation
    # =========================================================================
    print("6. Testing Submission Generation...")
    generate_submission(model, test_loader, device)

    submission_path = os.path.join(Config.submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"   Submission shape: {sub_df.shape}")

    # Check if IDs match the test loader subset
    # Since we used debug mode, the test loader is subsampled.
    # The generate_submission function loads the FULL test.csv from metadata to get IDs.
    # However, in debug mode, generate_submission logic in engine.py handles truncation:
    # "if Config.debug: test_df = test_df.iloc[: len(test_scores)]"
    # So the submission file should have rows equal to Config.debug_sample_size

    assert (
        len(sub_df) == Config.debug_sample_size
    ), f"Submission length {len(sub_df)} does not match debug sample size {Config.debug_sample_size}"

    # Verify columns
    assert "id" in sub_df.columns and "score" in sub_df.columns
    assert sub_df["score"].dtype == float or sub_df["score"].dtype == np.float64

    print("--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
