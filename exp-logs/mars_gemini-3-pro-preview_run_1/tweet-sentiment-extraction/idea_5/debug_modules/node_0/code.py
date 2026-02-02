import sys
import os
import torch
import numpy as np
import warnings
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import get_loaders, get_gaussian_target
from library.model import SentimentModel
from library.awp import AWP
from library.engine import train_fn, eval_fn, predict_fn, get_optimizer_params

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=== Starting Sentiment Extraction Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Set seed for reproducibility
    seed_everything(Config.seed)

    # Override Config for speed and demonstration purposes
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.print_freq = 5  # Print every 5 steps
    Config.awp_start_epoch = 0  # Enable AWP immediately for testing
    Config.train_on_neutral = False  # Keep default strategy

    device = Config.device
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Jaccard metric
    s1, s2 = "hello world", "hello world"
    assert jaccard(s1, s2) == 1.0, "Jaccard identity check failed"
    assert jaccard("hello world", "hello") > 0.0, "Jaccard partial match check failed"

    # Test Gaussian Target Generation (Soft Labels)
    target_dist = get_gaussian_target(index=5, length=10, sigma=1.0)
    assert len(target_dist) == 10, "Gaussian target length mismatch"
    assert np.argmax(target_dist) == 5, "Gaussian peak index mismatch"
    assert np.isclose(target_dist.sum(), 1.0), "Gaussian distribution does not sum to 1"

    print("Utilities verified successfully.")

    # ---------------------------------------------------------
    # 3. Data Loading (Debug Mode)
    # ---------------------------------------------------------
    print("\n[3] Loading Data (Debug Mode)...")

    # debug=True loads a tiny subset (100 train, 50 val/test)
    # load_cached_data=False forces reprocessing to test the pipeline
    train_loader, val_loader, test_loader = get_loaders(
        debug=True, load_cached_data=False
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "span_masks",
    ]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    print(f"Train Batch Shape (Input IDs): {batch['input_ids'].shape}")
    assert batch["input_ids"].shape[0] == Config.train_batch_size, "Batch size mismatch"
    print("Data loading verified.")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing Model...")
    model = SentimentModel()
    model.to(device)
    print("Model initialized successfully.")

    # ---------------------------------------------------------
    # 5. Forward Pass Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Forward Pass...")
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    start_logits, end_logits, aux_logits = model(input_ids, mask)

    # Check output shapes
    # Logits should be (Batch, Max_Len)
    expected_shape = (Config.train_batch_size, Config.max_len)
    assert (
        start_logits.shape == expected_shape
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert (
        end_logits.shape == expected_shape
    ), f"End logits shape mismatch: {end_logits.shape}"
    assert (
        aux_logits.shape == expected_shape
    ), f"Aux logits shape mismatch: {aux_logits.shape}"
    print("Forward pass shapes verified.")

    # ---------------------------------------------------------
    # 6. Adversarial Weight Perturbation (AWP) Check
    # ---------------------------------------------------------
    print("\n[6] Verifying AWP Logic...")

    # Create a dummy optimizer just for this check
    optimizer_dummy = AdamW(model.parameters(), lr=1e-3)
    awp = AWP(model, optimizer_dummy, adv_lr=0.1, adv_eps=0.1, start_epoch=0)

    # Generate gradients via a dummy backward pass
    dummy_loss = start_logits.mean()
    dummy_loss.backward(retain_graph=True)

    # Track a specific parameter to verify perturbation
    param_name = "span_head.weight"
    original_weight = model.span_head.weight.data.clone()

    # Attack (Perturb weights)
    awp.attack()
    perturbed_weight = model.span_head.weight.data.clone()

    # Verify weights changed
    diff = torch.norm(original_weight - perturbed_weight).item()
    assert diff > 0, "AWP failed to perturb weights"
    print(f"AWP Perturbation Magnitude: {diff:.6f}")

    # Restore weights
    awp._restore()
    restored_weight = model.span_head.weight.data

    # Verify weights restored
    assert torch.allclose(
        original_weight, restored_weight
    ), "AWP failed to restore weights"
    print("AWP logic verified.")

    # Clean up gradients
    optimizer_dummy.zero_grad()

    # ---------------------------------------------------------
    # 7. Training Loop Integration
    # ---------------------------------------------------------
    print("\n[7] Running Training Loop (1 Epoch)...")

    # Setup Optimizer with LLRD (Layer-wise Learning Rate Decay)
    optimizer_params = get_optimizer_params(model, encoder_lr=2e-5, decoder_lr=2e-5)
    optimizer = AdamW(
        optimizer_params, lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Setup Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Execute Training Function
    avg_loss = train_fn(train_loader, model, optimizer, 0, scheduler, device)
    print(f"Training Epoch Completed. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss resulted in NaN"

    # ---------------------------------------------------------
    # 8. Evaluation Integration
    # ---------------------------------------------------------
    print("\n[8] Running Evaluation on Validation Set...")
    val_jaccard = eval_fn(val_loader, model, device)
    print(f"Validation Jaccard Score: {val_jaccard:.4f}")
    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score is out of valid range [0, 1]"

    # ---------------------------------------------------------
    # 9. Prediction Integration
    # ---------------------------------------------------------
    print("\n[9] Running Prediction on Test Set...")
    predictions = predict_fn(test_loader, model, device)

    print(f"Generated {len(predictions)} predictions.")

    # Verify predictions
    assert len(predictions) > 0, "No predictions were generated"
    assert isinstance(
        predictions[0], str
    ), "Prediction output format is incorrect (expected string)"

    # Check count against dataset size (Debug mode loads 50 test samples)
    expected_count = len(test_loader.dataset)
    assert (
        len(predictions) == expected_count
    ), f"Prediction count mismatch: Expected {expected_count}, got {len(predictions)}"

    print("Prediction pipeline verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
