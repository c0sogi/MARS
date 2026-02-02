import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CdiscountDataset, collate_flatten
from library.model import get_model
from library.engine import get_idx_to_id_map


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create a small training subset to ensure execution finishes within 10 minutes
    print("Creating fast training subset...")
    full_train_df = pd.read_csv(Config.TRAIN_META)
    subset_train_df = full_train_df.sample(n=5000, random_state=Config.SEED)
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_train_df.to_csv(subset_train_path, index=False)

    # Override Config for this run
    Config.TRAIN_META = subset_train_path
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 512  # High batch size for A100 efficiency

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_dataset = CdiscountDataset(Config.TRAIN_META, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_flatten,
        drop_last=True,
    )

    # Must validate on the entire hold-out set
    val_dataset = CdiscountDataset(Config.VAL_META, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Inference can handle larger batches
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_flatten,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = get_model(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler()

    # 4. Training Loop (Fast Baseline)
    print("Starting Training...")
    model.train()
    for i, (images, targets, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=True):
            output = model(images)
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 10 == 0:
            print(f"Train Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

    # 5. Validation and Failure Analysis
    print("Starting Validation on entire hold-out set...")
    model.eval()

    val_correct = []
    val_pids = []
    total_correct = 0
    total_samples = 0

    # Load metadata for failure analysis mapping
    val_meta_df = pd.read_csv(Config.VAL_META)
    # Map product_id to bson_length to analyze if size correlates with error
    pid_to_len = dict(zip(val_meta_df.product_id, val_meta_df.bson_length))

    with torch.no_grad():
        for i, (images, targets, pids) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=True):
                output = model(images)

            # Calculate Top-1 Accuracy
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            # Store results
            correct_np = correct.view(-1).cpu().numpy().astype(int)
            pids_np = pids.numpy()

            val_correct.extend(correct_np)
            val_pids.extend(pids_np)

            total_correct += correct_np.sum()
            total_samples += len(correct_np)

            if i % 100 == 0:
                print(f"Val Batch {i}/{len(val_loader)}")

    final_acc = (total_correct / total_samples) * 100.0
    print(f"Final Validation Metric: {final_acc:.10f}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    results_df = pd.DataFrame({"product_id": val_pids, "is_correct": val_correct})

    # Map features (BSON length)
    results_df["bson_length"] = results_df["product_id"].map(pid_to_len)
    results_df["error"] = 1 - results_df["is_correct"]

    # Calculate Correlation
    # We drop NaNs just in case of mapping issues, though ids should match exactly
    results_df = results_df.dropna(subset=["bson_length"])
    correlation = results_df["error"].corr(results_df["bson_length"])
    print(f"Correlation between Error and BSON Length: {correlation:.10f}")

    # 6. Submission (Conditional)
    if final_acc > 61.16:
        print("Validation metric > 61.16. Generating submission...")
        test_dataset = CdiscountDataset(Config.TEST_META, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            collate_fn=collate_flatten,
        )

        all_probs = []
        all_pids = []

        print("Running Inference on Test Set...")
        with torch.no_grad():
            for i, (images, _, product_ids) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)

                with autocast(enabled=True):
                    output = model(images)
                    probs = torch.softmax(output, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_pids.append(product_ids.numpy())

                if i % 100 == 0:
                    print(f"Test Batch {i}/{len(test_loader)}")

        # Aggregation (Late Fusion)
        flat_probs = np.concatenate(all_probs, axis=0)
        flat_pids = np.concatenate(all_pids, axis=0)

        # Sort by product_id
        sort_idx = np.argsort(flat_pids)
        sorted_pids = flat_pids[sort_idx]
        sorted_probs = flat_probs[sort_idx]

        # Sum probabilities per product
        unique_pids, indices = np.unique(sorted_pids, return_index=True)
        summed_probs = np.add.reduceat(sorted_probs, indices, axis=0)
        final_preds_idx = np.argmax(summed_probs, axis=1)

        # Map indices back to category_ids
        idx_to_id = get_idx_to_id_map()
        max_idx = max(idx_to_id.keys())
        lookup_table = np.zeros(max_idx + 1, dtype=np.int64)
        for idx, cat_id in idx_to_id.items():
            lookup_table[idx] = cat_id

        final_category_ids = lookup_table[final_preds_idx]

        submission = pd.DataFrame(
            {"_id": unique_pids, "category_id": final_category_ids}
        )
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation Metric {final_acc:.4f} <= 61.16. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
