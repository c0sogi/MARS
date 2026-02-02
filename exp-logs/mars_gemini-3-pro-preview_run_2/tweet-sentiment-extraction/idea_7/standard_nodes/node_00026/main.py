import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import library components
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import TweetDataset, process_data, get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run():
    # ====================================================
    # 1. Setup & Configuration
    # ====================================================
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We use the metadata/train.csv (80% of data) for training to strictly separate
    # the hold-out validation set (metadata/val.csv)
    Config.TRAIN_CSV = "./metadata/train.csv"
    Config.EPOCHS = 2  # Reduced epochs for 2-hour time limit

    # Ensure output directory exists
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)

    # Clear cache to force reprocessing with the new TRAIN_CSV
    cache_path = os.path.join(Config.OUTPUT_DIR, "train_folds.parquet")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    print(f"Starting execution with {Config.N_FOLDS}-Fold CV on {Config.TRAIN_CSV}")
    print(f"Training for {Config.EPOCHS} epochs per fold.")

    # ====================================================
    # 2. Data Processing
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Generate folds on the training subset
    _ = process_data(load_cached_data=False)

    # ====================================================
    # 3. Training Loop
    # ====================================================
    device = Config.DEVICE
    trained_model_paths = []

    for fold in range(Config.N_FOLDS):
        print(f"\nTraining Fold {fold + 1}/{Config.N_FOLDS}...")

        # Get Dataloaders
        train_loader, val_loader = get_loaders(fold, tokenizer, load_cached_data=True)

        # Initialize Model
        model = TweetModel()
        model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Training
        best_jaccard = -1
        model_save_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            val_loss, val_jaccard = eval_fn(val_loader, model, device)

            print(
                f"  Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            # Save best model for this fold
            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), model_save_path)

        trained_model_paths.append(model_save_path)

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # ====================================================
    # 4. Hold-out Validation (Ensemble Inference)
    # ====================================================
    print("\nRunning Validation on Hold-out Set (metadata/val.csv)...")

    # Load Hold-out Data
    val_df = pd.read_csv("./metadata/val.csv")
    val_df.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)
    val_df["text"] = val_df["text"].astype(str)
    val_df["selected_text"] = val_df["selected_text"].astype(str)

    val_dataset = TweetDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    def run_inference(loader, model_paths):
        """
        Runs ensemble inference on a dataloader.
        Returns predictions, texts, and sentiments.
        """
        # Load models
        models = []
        for path in model_paths:
            m = TweetModel()
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device)
            m.eval()
            models.append(m)

        preds = []

        with torch.no_grad():
            for data in loader:
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

                # Metadata for reconstruction
                orig_texts = data["orig_text"]
                sentiments = data["sentiment"]
                offsets = data["offsets"].cpu().numpy()

                # Ensemble Logits
                avg_start = None
                avg_end = None

                for m in models:
                    s, e = m(ids, mask, token_type_ids)
                    if avg_start is None:
                        avg_start = s
                        avg_end = e
                    else:
                        avg_start += s
                        avg_end += e

                avg_start /= len(models)
                avg_end /= len(models)

                # Probabilities
                start_probs = torch.softmax(avg_start, dim=1).cpu().numpy()
                end_probs = torch.softmax(avg_end, dim=1).cpu().numpy()

                # Decode
                for i in range(len(ids)):
                    text = orig_texts[i]
                    sentiment = sentiments[i]
                    offset = offsets[i]

                    p_start = np.argmax(start_probs[i])
                    p_end = np.argmax(end_probs[i])

                    if p_start > p_end:
                        p_end = p_start

                    # Neutral Heuristic
                    if sentiment == "neutral":
                        pred_text = text
                    else:
                        if p_start < len(offset) and p_end < len(offset):
                            start_char = offset[p_start][0]
                            end_char = offset[p_end][1]
                            pred_text = text[start_char:end_char]
                        else:
                            pred_text = text

                    preds.append(pred_text)

        # Cleanup
        for m in models:
            del m
        torch.cuda.empty_cache()

        return preds

    # Generate Predictions
    val_preds = run_inference(val_loader, trained_model_paths)

    # Compute Metric
    scores = [jaccard(p, t) for p, t in zip(val_preds, val_df["selected_text"])]
    final_metric = np.mean(scores)

    print(f"Final Validation Metric: {final_metric}")

    # ====================================================
    # 5. Failure Analysis
    # ====================================================
    print("\nPerforming Failure Analysis...")
    val_df["error"] = 1.0 - np.array(scores)
    val_df["text_len"] = val_df["text"].apply(len)

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    val_df["sentiment_numeric"] = val_df["sentiment"].map(sentiment_map)

    # Compute Correlations (using numpy)
    corr_len = np.corrcoef(val_df["error"], val_df["text_len"])[0, 1]
    corr_sent = np.corrcoef(val_df["error"], val_df["sentiment_numeric"])[0, 1]

    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Sentiment): {corr_sent:.4f}")

    # ====================================================
    # 6. Submission
    # ====================================================
    if final_metric > 0.7205:
        print("\nMetric threshold met (> 0.7205). Generating submission...")

        test_df = pd.read_csv(Config.TEST_CSV)
        test_df["text"] = test_df["text"].astype(str)

        test_dataset = TweetDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = run_inference(test_loader, trained_model_paths)

        submission = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": test_preds}
        )

        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"Metric {final_metric} did not meet threshold. Submission skipped.")


if __name__ == "__main__":
    run()
