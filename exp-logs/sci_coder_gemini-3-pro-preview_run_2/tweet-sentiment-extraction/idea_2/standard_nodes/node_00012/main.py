import os
import sys
import pandas as pd
import numpy as np
import torch
import csv
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import TweetDataset
from library.model import TweetModel
from library.engine import run_training, eval_fn, generate_submission
from library.utils import seed_everything, jaccard


def run_failure_analysis(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates error magnitude (1 - Jaccard) and correlates it with input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    results = []

    # Create a map for ground truth for fast lookup
    gt_map = dict(
        zip(val_df["textID"].astype(str), val_df["selected_text"].astype(str))
    )

    with torch.no_grad():
        for d in val_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            offsets = d["offsets"].numpy()
            ids = d["textID"]
            texts = d["text"]
            sentiments = d["sentiment"]

            for i in range(len(ids)):
                text_id = ids[i]
                text = texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]
                selected_text = gt_map.get(text_id, "")

                # Prediction logic mirroring eval_fn
                if sentiment == "neutral":
                    pred_string = text
                else:
                    start_p = start_probs[i]
                    end_p = end_probs[i]
                    score_mat = start_p[:, None] + end_p[None, :]
                    upper_tri_mask = np.triu(np.ones_like(score_mat))
                    masked_score = score_mat * upper_tri_mask
                    masked_score[upper_tri_mask == 0] = -1e9

                    max_idx = np.unravel_index(
                        np.argmax(masked_score), masked_score.shape
                    )
                    idx_start, idx_end = max_idx

                    try:
                        char_start = offset[idx_start][0]
                        char_end = offset[idx_end][1]
                        if char_start == 0 and char_end == 0:
                            pred_string = text
                        else:
                            pred_string = text[char_start:char_end]
                    except:
                        pred_string = text

                score = jaccard(selected_text, pred_string)

                # Store results for analysis
                results.append(
                    {
                        "text_len": len(str(text).split()),
                        "sentiment": sentiment,
                        "jaccard": score,
                        "error": 1.0 - score,
                    }
                )

    df_results = pd.DataFrame(results)

    # Calculate and print correlation
    corr = df_results["error"].corr(df_results["text_len"])
    print(f"Correlation (Error vs Input Length): {corr}")

    # Calculate and print mean error by sentiment
    print("Mean Error by Sentiment:")
    print(df_results.groupby("sentiment")["error"].mean())


def main():
    # 1. Configuration
    # Upgrading to roberta-large with tuned hyperparameters (Cite solution_lesson_node_00010)
    config = Config(
        epochs=5,
        train_batch_size=16,
        valid_batch_size=32,
        model_name="roberta-large",
        learning_rate=2e-5,
        seed=42,
    )
    seed_everything(config.SEED)
    config.NUM_WORKERS = 0  # Fix: Disable multiprocessing to prevent Bus Error

    print("Initializing Data and Model...")

    # 2. Data Loading
    # Load cached data if available to save time
    train_dataset = TweetDataset("train", config, load_cached_data=True)
    val_dataset = TweetDataset("val", config, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimizer
    model = TweetModel(config)
    model.to(config.DEVICE)

    # Weight decay setup
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)

    # Scheduler
    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # 4. Training
    val_df = pd.read_csv(config.VAL_PATH)

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        num_epochs=config.EPOCHS,
        save_path=config.MODEL_SAVE_PATH,
        val_df=val_df,
        patience=2,
    )

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    else:
        print("Warning: Best model file not found. Using current model state.")

    model.to(config.DEVICE)
    model.eval()

    val_loss, val_jaccard = eval_fn(val_loader, model, config.DEVICE, val_df)
    print(f"Final Validation Metric: {val_jaccard}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, val_df, config.DEVICE)

    # 7. Submission
    THRESHOLD = 0.7061030713654458
    if val_jaccard > THRESHOLD:
        print("\nMetric exceeds threshold. Generating submission...")
        test_dataset = TweetDataset("test", config, load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate initial CSV using engine
        generate_submission(test_loader, model, config.DEVICE, config.SUBMISSION_PATH)

        # Post-process to ensure strict quoting as requested (e.g., 2,"very good")
        # Standard pandas to_csv might not quote simple strings by default without QUOTE_NONNUMERIC
        try:
            df_sub = pd.read_csv(config.SUBMISSION_PATH)
            # Force quoting for non-numeric fields (textID is usually treated as string/object here, selected_text is string)
            # We explicitly quote everything to be safe and match the format 'textID,"selected_text"' roughly
            df_sub.to_csv(
                config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC
            )
            print(
                f"Submission re-saved with strict quoting to {config.SUBMISSION_PATH}"
            )
        except Exception as e:
            print(f"Error during submission post-processing: {e}")

    else:
        print(
            f"\nValidation metric {val_jaccard} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
