import os
import torch
import pandas as pd
import numpy as np
import warnings
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.dataset import get_data
from library.model import TweetModel
from library.loss import HybridLoss
from library.engine import train_fn, eval_fn
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Tweet Sentiment Extraction Demo ===")

    # 1. Configuration Overrides for Speed and Demo Purposes
    print("\n[1] Configuring environment...")
    Config.debug = True  # Use small subset (500 samples)
    Config.epochs = 1  # Run only 1 epoch
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.num_workers = 2

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.seed)
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Device: {Config.device}")

    # 2. Data Loading
    print("\n[2] Loading and Processing Data...")
    # Force load_cached_data=False to ensure we process the debug subset correctly
    # and verify the tokenization logic.
    train_ds, val_ds, test_ds = get_data(load_cached_data=False)

    print(f"    Train Dataset Size: {len(train_ds)}")
    print(f"    Val Dataset Size:   {len(val_ds)}")
    print(f"    Test Dataset Size:  {len(test_ds)}")

    # Verify dataset size matches debug limit
    assert (
        len(train_ds) <= Config.debug_sample_size
    ), "Train dataset size exceeds debug limit"

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = TweetModel()
    model.to(Config.device)

    # Verification: Dummy Forward Pass
    print("    Verifying forward pass shapes...")
    dummy_batch = next(iter(train_loader))
    input_ids = dummy_batch["ids"].to(Config.device)
    mask = dummy_batch["mask"].to(Config.device)
    token_type_ids = dummy_batch["token_type_ids"].to(Config.device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask, token_type_ids)

    # Check shapes: (Batch_Size, Seq_Len)
    assert start_logits.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Shape mismatch: {end_logits.shape}"
    print("    Forward pass check passed.")

    # 4. Training Setup
    print("\n[4] Setting up Optimizer and Loss...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    criterion = HybridLoss()

    num_train_steps = int(len(train_ds) / Config.train_batch_size * Config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    print("\n[5] Training for 1 Epoch...")
    avg_loss = train_fn(
        train_loader, model, optimizer, Config.device, scheduler, criterion
    )
    print(f"    Epoch 1 Training Loss: {avg_loss:.4f}")

    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss > 0, "Training loss should be positive"

    # 6. Evaluation Loop (Validation)
    print("\n[6] Evaluating on Validation Set...")
    val_jaccard = eval_fn(val_loader, model, Config.device)
    print(f"    Validation Jaccard Score: {val_jaccard:.4f}")

    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score out of range [0, 1]"

    # 7. Inference on Test Set (Custom Loop)
    # Note: We cannot use engine.eval_fn for test set because it expects ground truth 'orig_selected'
    print("\n[7] Running Inference on Test Set...")
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for data in test_loader:
            input_ids = data["ids"].to(Config.device, dtype=torch.long)
            attention_mask = data["mask"].to(Config.device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(Config.device, dtype=torch.long)

            # Metadata
            orig_tweets = data["orig_tweet"]
            sentiments = data["sentiment"]
            offsets = data["offsets"].numpy()

            # Forward
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            # Decode
            for i in range(len(input_ids)):
                tweet = orig_tweets[i]
                sentiment = sentiments[i]
                offset = offsets[i]
                s_logits = start_logits[i]
                e_logits = end_logits[i]

                # Neutral Heuristic
                if Config.neutral_heuristic and sentiment == "neutral":
                    pred_text = tweet
                else:
                    scores = s_logits[:, np.newaxis] + e_logits[np.newaxis, :]
                    upper_tri_mask = np.triu(np.ones_like(scores), k=0)
                    scores = np.where(upper_tri_mask == 1, scores, -np.inf)

                    max_idx = np.argmax(scores)
                    idx_start, idx_end = np.unravel_index(max_idx, scores.shape)

                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    pred_text = tweet[char_start:char_end]

                predictions.append(pred_text)
                # We don't have IDs in the dataset class output, but we can assume order matches
                # In a real scenario, we would include textID in the dataset __getitem__
                # Here we just verify prediction generation.

    # 8. Submission Generation Verification
    print("\n[8] Generating Submission File...")
    # Load original test csv to get IDs (since Dataset didn't return them)
    df_test = pd.read_csv(Config.TEST_META)
    if Config.debug:
        # If debug, we need the specific rows sampled.
        # However, dataset.get_data samples internally and doesn't return indices.
        # For this demo, we just check that we generated predictions for the processed samples.
        print(f"    Generated {len(predictions)} predictions.")
        assert len(predictions) == len(test_ds), "Prediction count mismatch"
    else:
        df_submission = pd.DataFrame(
            {"textID": df_test["textID"], "selected_text": predictions}
        )
        submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
        df_submission.to_csv(submission_path, index=False)
        print(f"    Submission saved to {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
