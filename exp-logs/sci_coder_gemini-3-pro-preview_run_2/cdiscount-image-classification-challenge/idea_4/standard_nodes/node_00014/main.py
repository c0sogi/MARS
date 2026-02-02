import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler

# Import provided library modules
from library.config import Config
from library.dataset import (
    BSONDataset,
    get_transforms,
    train_collate_fn,
    eval_collate_fn,
)
from library.model import get_model
from library.engine import (
    CategoryMapper,
    seed_everything,
    train_one_epoch,
    validate,
    predict,
)


def main():
    # 1. Configuration & Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust configuration for the time limit
    # Cite solution_lesson_node_00008: Prioritize data throughput (unique samples) over repetition.
    # Cite solution_lesson_node_00007: Use OneCycleLR with large data for few epochs (1 epoch).
    # Increased sample size to 3.2M (4x baseline) and reduced to 1 epoch.
    # With BATCH_SIZE=512, total optimizer steps remain ~6250 (same as baseline).
    TRAIN_SAMPLE_SIZE = 3200000
    EPOCHS = 1

    Config.update(epochs=EPOCHS)

    print(
        f"Configuration: Device={device}, Epochs={EPOCHS}, Train Sample Size={TRAIN_SAMPLE_SIZE}"
    )

    # 2. Data Preparation (Subsampling)
    print("Loading training metadata...")
    df_train = pd.read_csv(Config.TRAIN_META)

    if len(df_train) > TRAIN_SAMPLE_SIZE:
        print(f"Subsampling training data to {TRAIN_SAMPLE_SIZE} records...")
        df_train = df_train.sample(
            n=TRAIN_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    train_meta_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    df_train.to_csv(train_meta_path, index=False)

    # Update Config to use the subset
    Config.TRAIN_META = train_meta_path

    # 3. DataLoaders
    print("Initializing DataLoaders...")
    mapper = CategoryMapper()

    train_dataset = BSONDataset(
        Config.TRAIN_META,
        mode="train",
        transform=get_transforms("train", Config.IMG_SIZE),
    )

    # We use the full validation set for accurate metric calculation
    val_dataset = BSONDataset(
        Config.VAL_META, mode="val", transform=get_transforms("val", Config.IMG_SIZE)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=eval_collate_fn,
        pin_memory=True,
    )

    # 4. Model & Optimizer Setup
    print("Initializing Model...")
    model = get_model(num_classes=mapper.num_classes, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    scaler = GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # 5. Training Loop
    print("Starting training...")
    best_acc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, device, mapper
        )

        val_loss, val_acc = validate(model, val_loader, loss_fn, device, mapper)

        print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")
        print(f"Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"-> New best model saved! Acc: {best_acc:.6f}")

    # 6. Final Metric Reporting
    print(f"Final Validation Metric: {best_acc}")

    # 7. Failure Analysis
    print("\n==== Failure Analysis ====")
    print("Loading best model for analysis...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    results = []
    print("Running inference on validation set for analysis...")
    with torch.no_grad():
        for flat_images, labels, product_ids, counts in val_loader:
            flat_images = flat_images.to(device)
            targets = mapper.to_idx(labels).to(device)

            with torch.cuda.amp.autocast():
                logits = model(flat_images)
                probs = torch.softmax(logits, dim=1)

                # Late Fusion
                probs_split = torch.split(probs, counts.tolist())
                product_probs = torch.stack([p.mean(dim=0) for p in probs_split])

            _, preds = product_probs.max(1)

            # Collect correctness and image counts
            correct_mask = preds.eq(targets).cpu().numpy().astype(int)
            img_counts = counts.cpu().numpy()

            for c, n in zip(correct_mask, img_counts):
                results.append({"correct": c, "num_imgs": n})

    df_analysis = pd.DataFrame(results)

    # Calculate correlation
    correlation = df_analysis["correct"].corr(df_analysis["num_imgs"])
    print(f"Correlation between error (correctness) and image count: {correlation:.4f}")
    print(
        "Interpretation: Positive correlation implies products with more images are classified more accurately."
    )

    # 8. Submission
    if best_acc > 0.6116:
        print("\n==== Generating Submission ====")
        test_dataset = BSONDataset(
            Config.TEST_META,
            mode="test",
            transform=get_transforms("test", Config.IMG_SIZE),
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=eval_collate_fn,
            pin_memory=True,
        )

        df_submission = predict(model, test_loader, device, mapper)

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        df_submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nValidation metric ({best_acc:.4f}) is below threshold (0.6116). Skipping submission."
        )


if __name__ == "__main__":
    main()
