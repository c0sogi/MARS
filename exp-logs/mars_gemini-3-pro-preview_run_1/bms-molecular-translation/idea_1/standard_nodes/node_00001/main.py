import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import time

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import ChemicalDataset, get_transforms
from library.model import ShowAndTell
from library.trainer import Trainer
from library.utils import load_checkpoint, compute_levenshtein


def main():
    print("Initializing Pipeline...")

    # 1. Tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(load_cached_data=True)

    # 2. Data Loading & Sampling
    print("Loading Metadata...")
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sampling for Fast Baseline (Time Limit Constraint)
    # We use 30,000 training samples to ensure training finishes quickly.
    TRAIN_SAMPLE_SIZE = 30000
    # We use a small subset for internal validation during training loop to save time
    VAL_LOOP_SAMPLE_SIZE = 2000

    df_train = df_train_full.sample(
        n=min(len(df_train_full), TRAIN_SAMPLE_SIZE), random_state=Config.SEED
    ).reset_index(drop=True)
    df_val_loop = df_val_full.sample(
        n=min(len(df_val_full), VAL_LOOP_SAMPLE_SIZE), random_state=Config.SEED
    ).reset_index(drop=True)

    print(f"Training on {len(df_train)} samples.")
    print(f"Internal validation on {len(df_val_loop)} samples.")
    print(
        f"Final evaluation will be on full validation set: {len(df_val_full)} samples."
    )

    # 3. Datasets & Loaders
    train_dataset = ChemicalDataset(
        df_train, tokenizer, transform=get_transforms("train")
    )
    val_loop_dataset = ChemicalDataset(
        df_val_loop, tokenizer, transform=get_transforms("valid")
    )

    # Use num_workers=2 to avoid too much overhead but keep feeding GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loop_loader = DataLoader(
        val_loop_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Model Setup
    print("Setting up Model...")
    model = ShowAndTell(
        vocab_size=len(tokenizer),
        sos_idx=tokenizer.sos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
        max_len=Config.MAX_LEN,
    )
    model = model.to(Config.DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # CrossEntropyLoss expects flattened inputs. We ignore the padding token.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # 5. Training
    print("Starting Training...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loop_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        patience=3,  # Lower patience for baseline
    )

    # Run for limited epochs to fit in time
    trainer.fit(epochs=5)

    # 6. Final Validation on Full Hold-out Set
    print("\n--- Starting Final Validation on Full Hold-out Set ---")

    # Load best model
    epoch, best_metric = load_checkpoint(Config.MODEL_PATH, model)
    model.eval()

    # Create loader for full validation set
    # Use larger batch size for inference
    val_full_dataset = ChemicalDataset(
        df_val_full, tokenizer, transform=get_transforms("valid")
    )
    val_full_loader = DataLoader(
        val_full_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    levenshtein_scores = []
    ground_truth_lengths = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_full_loader):
            images = images.to(Config.DEVICE)

            # Inference
            preds = model.sample(images)
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.numpy()

            for j in range(len(images)):
                pred_str = tokenizer.decode(preds_cpu[j])
                target_str = tokenizer.decode(labels_cpu[j])

                score = compute_levenshtein(pred_str, target_str)
                levenshtein_scores.append(score)
                ground_truth_lengths.append(len(target_str))

            if i % 100 == 0 and i > 0:
                print(f"Processed {i * val_full_loader.batch_size} samples...")

    final_metric = np.mean(levenshtein_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(levenshtein_scores) > 1:
        corr, _ = pearsonr(ground_truth_lengths, levenshtein_scores)
        print(
            f"Correlation between Error (Levenshtein) and Ground Truth Length: {corr:.4f}"
        )
        if abs(corr) > 0.3:
            print(
                "Observation: Stronger correlation indicates the model struggles with longer/more complex molecules."
            )
        else:
            print(
                "Observation: Weak correlation suggesting errors are distributed across lengths."
            )

    # 8. Test Prediction & Submission
    print("\n--- Generating Submission ---")
    test_dataset = ChemicalDataset(df_test, tokenizer, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = []
    print("Running inference on test set...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(Config.DEVICE)
            preds = model.sample(images)
            preds_cpu = preds.cpu().numpy()

            for j in range(len(images)):
                pred_str = tokenizer.decode(preds_cpu[j])
                predictions.append(pred_str)

            if i % 100 == 0 and i > 0:
                print(f"Processed {i * test_loader.batch_size} test samples...")

    # Create submission dataframe
    df_sub = pd.DataFrame({"image_id": df_test["image_id"], "InChI": predictions})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generated successfully.")


if __name__ == "__main__":
    main()
