import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, clean_text, get_score
from library.dataset import InsultDataset, load_dataset_dataframe, get_tokenizer
from library.model import InsultModel
from library.awp import AWP
from library.trainer import train_one_epoch, valid_one_epoch, get_optimizer_params
from library.inference import predict_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Insult Detection Library Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.model_backbones = [
        "prajjwal1/bert-tiny"
    ]  # Tiny model for CPU/fast execution
    Config.epochs = 1
    Config.n_folds = 2  # We will only simulate fold 0
    Config.batch_size = 4
    Config.max_len = 32  # Short sequence length for speed
    Config.working_dir = "./working/demo_run"
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")
    Config.debug = True

    # Ensure working directory exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seed
    seed_everything(Config.seed)
    print("Configuration updated. Random seed set.")

    # =========================================================================
    # 2. Utility Verification
    # =========================================================================
    print("\n[2] Verifying Utilities...")

    # Test clean_text
    raw_text = '"User\\nComment with \\"quotes\\""'
    cleaned = clean_text(raw_text)
    print(f"Raw: {raw_text} -> Cleaned: {cleaned}")
    # The cleaner handles unicode escapes and quotes.
    # Note: The exact output depends on python's ast.literal_eval behavior on the string
    assert isinstance(cleaned, str), "clean_text should return a string"

    # Test get_score
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    score = get_score(y_true, y_pred)
    print(f"AUC Score check: {score}")
    assert 0 <= score <= 1, "AUC score must be between 0 and 1"

    # =========================================================================
    # 3. Data Loading & Dataset
    # =========================================================================
    print("\n[3] Preparing Data...")

    # Create a small subset of training data for the demo
    full_train_df = pd.read_csv(Config.train_path)
    subset_train_df = full_train_df.head(20).copy()  # Only 20 samples
    subset_train_path = os.path.join(Config.working_dir, "train_subset.csv")
    subset_train_df.to_csv(subset_train_path, index=False)

    # Load using library function (tests caching logic)
    df_train = load_dataset_dataframe(subset_train_path, "train_subset_cache")
    assert len(df_train) == 20, "Dataframe loading failed size check"

    # Initialize Tokenizer and Dataset
    model_name = Config.model_backbones[0]
    tokenizer = get_tokenizer(model_name)

    train_dataset = InsultDataset(df_train, tokenizer, Config.max_len)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True
    )

    # Verify Batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target" in batch
    assert batch["input_ids"].shape == (Config.batch_size, Config.max_len)
    print("Dataset and DataLoader verified.")

    # =========================================================================
    # 4. Model & AWP
    # =========================================================================
    print("\n[4] Initializing Model and AWP...")

    device = Config.device
    model = InsultModel(model_name, Config)
    model.to(device)

    # Verify Forward Pass
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, mask)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.batch_size, 1), "Output shape mismatch"

    # Verify AWP (Adversarial Weight Perturbation)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    awp = AWP(
        model, optimizer, adv_lr=0.1, adv_eps=0.1
    )  # High LR to ensure visible change

    # Save original weight of a specific parameter
    param_name = "fc.weight"
    orig_weight = model.fc.weight.data.clone()

    # Simulate a step with gradients
    outputs = model(input_ids, mask)
    loss = torch.nn.BCEWithLogitsLoss()(outputs.view(-1), batch["target"].to(device))
    loss.backward()

    # Attack
    awp.attack()
    perturbed_weight = model.fc.weight.data.clone()

    # Check if weights changed
    diff = torch.norm(perturbed_weight - orig_weight).item()
    print(f"AWP Perturbation Norm: {diff:.6f}")
    assert diff > 0, "AWP did not perturb weights"

    # Restore
    awp.restore()
    restored_weight = model.fc.weight.data.clone()
    restore_diff = torch.norm(restored_weight - orig_weight).item()
    assert restore_diff < 1e-6, "AWP restore failed"
    print("AWP logic verified.")

    # =========================================================================
    # 5. Training Loop Simulation
    # =========================================================================
    print("\n[5] Simulating Training Loop (1 Epoch)...")

    # Re-initialize optimizer/scheduler for clean training
    optimizer_params = get_optimizer_params(
        model, encoder_lr=Config.lr, decoder_lr=Config.lr
    )
    optimizer = AdamW(optimizer_params, lr=Config.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 20)

    # Train one epoch
    train_loss = train_one_epoch(
        1, model, train_loader, optimizer, scheduler, device, awp=awp
    )

    # Validation (using same loader for demo simplicity)
    val_loss, val_auc = valid_one_epoch(model, train_loader, device)

    print(f"Training complete. Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Save model weights manually to simulate what run_fold does
    # Format expected by inference: {model_name_safe}_fold_{fold}.pth
    model_name_safe = model_name.replace("/", "_")
    fold = 0
    save_path = os.path.join(Config.working_dir, f"{model_name_safe}_fold_{fold}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # =========================================================================
    # 6. Inference Pipeline
    # =========================================================================
    print("\n[6] Running Inference Pipeline...")

    # Create dummy test data
    test_df = pd.DataFrame(
        {
            "Comment": ['"You are amazing"', '"You are terrible"', '"Neutral comment"'],
            "Date": ["20220101", "20220102", "20220103"],
        }
    )
    test_path = os.path.join(Config.working_dir, "test_demo.csv")
    test_df.to_csv(test_path, index=False)

    # Create dummy sample submission
    sample_sub = pd.DataFrame(
        {"Insult": [0, 0, 0], "Date": test_df["Date"], "Comment": test_df["Comment"]}
    )
    sample_sub_path = os.path.join(Config.working_dir, "sample_submission.csv")
    sample_sub.to_csv(sample_sub_path, index=False)

    # Run predict_fn
    # Note: We only saved fold 0. The inference loop iterates Config.n_folds.
    # We set Config.n_folds = 2 earlier. It will try to load fold 1, warn/skip, and average fold 0.
    predict_fn(
        test_path=test_path,
        submission_input_path=sample_sub_path,
        submission_output_path=Config.submission_path,
        model_dir=Config.working_dir,
        batch_size=2,
        device=device,
    )

    # Verify Submission
    if os.path.exists(Config.submission_path):
        sub_df = pd.read_csv(Config.submission_path)
        print("\nSubmission File Generated:")
        print(sub_df)
        assert len(sub_df) == 3, "Submission length mismatch"
        assert "Insult" in sub_df.columns, "Insult column missing"
        assert sub_df["Insult"].dtype == float, "Prediction type mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
