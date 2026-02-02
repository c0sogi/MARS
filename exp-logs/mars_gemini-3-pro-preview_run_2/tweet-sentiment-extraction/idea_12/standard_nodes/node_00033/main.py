import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data_factory import get_loaders
from library.modeling import TweetModel
from library.training_engine import train_fn, eval_fn
import library.inference_engine as inference_engine


def run():
    # --- 1. Configuration Overrides for Fast Baseline ---
    # Limit runtime to ensure completion within 2 hours
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 16  # A100 allows larger batches
    Config.DEBUG = False  # Use full dataset for valid results

    # Optimize for fixed input sizes
    torch.backends.cudnn.benchmark = True

    print("--- Starting Orchestration ---")
    print(f"Device: {Config.DEVICE}")
    print(f"Models: {Config.MODEL_BACKBONES}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.TRAIN_BATCH_SIZE}")

    # Ensure artifacts directory exists
    os.makedirs(Config.ARTIFACTS_DIR, exist_ok=True)

    # --- 2. Training Loop ---
    for model_name in Config.MODEL_BACKBONES:
        safe_name = model_name.replace("/", "_")
        print(f"\n===== Training Architecture: {model_name} =====")

        # Load Data (Cached)
        # Note: get_loaders handles caching. We load once per architecture.
        train_loader, val_loader, _ = get_loaders(
            model_name, batch_size=Config.TRAIN_BATCH_SIZE, load_cached_data=True
        )

        # Train 5 models (Simulated Folds via Seed Variation)
        for fold in range(Config.N_FOLDS):
            print(f"\n--- Fold {fold} / {Config.N_FOLDS - 1} ---")

            # Vary seed to create ensemble diversity (Bagging)
            current_seed = Config.SEED + fold
            seed_everything(current_seed)

            # Initialize Model
            model = TweetModel(model_name)
            model.to(Config.DEVICE)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            num_train_steps = len(train_loader) * Config.EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * num_train_steps),
                num_training_steps=num_train_steps,
            )

            # Training Loop
            model_save_path = os.path.join(
                Config.ARTIFACTS_DIR, f"{safe_name}_fold_{fold}.pth"
            )

            for epoch in range(Config.EPOCHS):
                train_loss = train_fn(
                    train_loader, model, optimizer, Config.DEVICE, scheduler
                )
                # Quick check on validation set (single model performance)
                val_loss, val_jaccard = eval_fn(val_loader, model, Config.DEVICE)

                print(
                    f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
                )

            # Save the model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            # Cleanup to free memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # --- 3. Ensemble Validation ---
    print("\n===== Performing Ensemble Validation =====")

    # Load validation metadata for ground truth
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Initialize accumulators for ensemble predictions
    val_results = {}
    for _, row in df_val.iterrows():
        text_content = str(row["text"])
        if text_content == "nan":
            text_content = ""
        t_len = len(text_content)
        val_results[row["textID"]] = {
            "start": np.zeros(t_len, dtype=np.float32),
            "end": np.zeros(t_len, dtype=np.float32),
            "text": text_content,
            "sentiment": str(row["sentiment"]),
            "target": (
                str(row["selected_text"]) if pd.notna(row["selected_text"]) else ""
            ),
        }

    # Iterate over all trained models to accumulate probabilities
    for model_name in Config.MODEL_BACKBONES:
        safe_name = model_name.replace("/", "_")

        # We need the loader again to get tokens/offsets for this specific tokenizer
        _, val_loader, _ = get_loaders(
            model_name, batch_size=Config.VALID_BATCH_SIZE, load_cached_data=True
        )

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(
                Config.ARTIFACTS_DIR, f"{safe_name}_fold_{fold}.pth"
            )
            if not os.path.exists(model_path):
                continue

            print(f"Aggregating predictions from: {safe_name}_fold_{fold}")

            model = TweetModel(model_name)
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)
            model.eval()

            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["ids"].to(Config.DEVICE)
                    mask = batch["mask"].to(Config.DEVICE)
                    tt_ids = batch["token_type_ids"].to(Config.DEVICE)
                    text_ids = batch["textID"]
                    offsets = batch["offsets"].numpy()

                    # Forward pass
                    start_logits, end_logits = model(ids, mask, tt_ids)

                    # Mask padding
                    active_mask = mask.bool()
                    start_logits[~active_mask] = -1e9
                    end_logits[~active_mask] = -1e9

                    # Softmax
                    start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                    end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                    # Accumulate to character level
                    for i, tid in enumerate(text_ids):
                        if tid not in val_results:
                            continue
                        res = val_results[tid]
                        txt_len = len(res["text"])

                        s_p = start_probs[i]
                        e_p = end_probs[i]
                        offs = offsets[i]

                        for tok_idx, (start_char, end_char) in enumerate(offs):
                            if start_char == 0 and end_char == 0:
                                continue

                            if start_char < txt_len:
                                res["start"][start_char] += s_p[tok_idx]

                            target_end_idx = end_char - 1
                            if target_end_idx >= 0 and target_end_idx < txt_len:
                                res["end"][target_end_idx] += e_p[tok_idx]

            del model
            torch.cuda.empty_cache()

    # Decode and Compute Metric
    total_jaccard = 0.0
    count = 0
    analysis_data = []

    for tid, data in val_results.items():
        text = data["text"]
        sentiment = data["sentiment"]
        target = data["target"]

        # Neutral Heuristic
        if sentiment == "neutral" or len(text) == 0:
            pred_text = text
        else:
            # Decode using sum of probabilities
            s_probs = data["start"]
            e_probs = data["end"]

            S = s_probs[:, None]
            E = e_probs[None, :]
            Score = S + E

            # Enforce start <= end
            tril_mask = np.triu(np.ones((len(text), len(text))), k=0)
            Score = Score * tril_mask
            Score[tril_mask == 0] = -1e9

            flat_idx = np.argmax(Score)
            best_start, best_end = np.unravel_index(flat_idx, Score.shape)

            pred_text = text[best_start : best_end + 1]

        score = jaccard(pred_text, target)
        total_jaccard += score
        count += 1

        analysis_data.append(
            {
                "jaccard": score,
                "error": 1.0 - score,
                "text_len": len(text.split()),
                "sentiment": sentiment,
            }
        )

    final_metric = total_jaccard / count if count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # --- 4. Failure Analysis ---
    print("\n===== Failure Analysis =====")
    df_analysis = pd.DataFrame(analysis_data)

    # Correlation
    corr_len = df_analysis["error"].corr(df_analysis["text_len"])
    print(f"Correlation (Error vs Input Length): {corr_len:.4f}")

    # Error by Sentiment
    print("Mean Error by Sentiment:")
    print(df_analysis.groupby("sentiment")["error"].mean())

    # --- 5. Submission ---
    if final_metric > 0.7205:
        print("\nMetric condition met (> 0.7205). Generating submission...")
        inference_engine.generate_submission(
            device=Config.DEVICE, load_cached_data=True
        )
    else:
        print(
            f"\nMetric condition NOT met ({final_metric} <= 0.7205). Skipping submission."
        )


if __name__ == "__main__":
    run()
