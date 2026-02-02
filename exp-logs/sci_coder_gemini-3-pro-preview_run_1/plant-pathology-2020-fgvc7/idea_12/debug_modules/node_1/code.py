import os
import sys
import pandas as pd
import torch
import torch.optim as optim
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_class_weights
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import DistillationLoss
from library.engine import fit, generate_submission


def run_demo():
    print("==== Starting Apple Disease Detection Pipeline Demo ====")

    # 1. Configuration
    # Using debug=True and epochs=1 for speed
    cfg = Config(debug=True, epochs=1)

    # Override batch size for the demo to ensure updates happen even with small data
    cfg.batch_size = 8

    # Ensure reproducibility
    seed_everything(cfg.seed)
    print("Configuration initialized and seeds set.")

    # 2. Data Loading
    print("\n[Data Loading]")
    # Load metadata
    if not os.path.exists(cfg.train_metadata_path):
        raise FileNotFoundError(f"Metadata not found at {cfg.train_metadata_path}")

    df_train = pd.read_csv(cfg.train_metadata_path)
    df_val = pd.read_csv(cfg.val_metadata_path)
    df_test = pd.read_csv(cfg.test_metadata_path)

    # Subsample for speed (Demo purposes)
    # We ensure we have enough samples for a couple of batches
    df_train_demo = df_train.head(32).reset_index(drop=True)
    df_val_demo = df_val.head(16).reset_index(drop=True)
    df_test_demo = df_test.head(16).reset_index(drop=True)

    print(f"Train samples: {len(df_train_demo)}")
    print(f"Val samples: {len(df_val_demo)}")
    print(f"Test samples: {len(df_test_demo)}")

    # Calculate Class Weights
    # We use the full df_train for weight calculation to be realistic,
    # even though we train on a subset here.
    class_weights = get_class_weights(
        df_train, cfg.target_cols, cache_dir=cfg.working_dir
    )
    print(f"Class weights calculated: {class_weights}")

    # 3. Dataset & DataLoader Initialization
    print("\n[Dataset Initialization]")

    # Transforms
    train_transform = get_transforms("train", cfg)
    val_transform = get_transforms("valid", cfg)

    # Datasets
    train_dataset = AppleDataset(
        df_train_demo, cfg, transform=train_transform, mode="standard"
    )
    val_dataset = AppleDataset(
        df_val_demo, cfg, transform=val_transform, mode="standard"
    )
    test_dataset = AppleDataset(df_test_demo, cfg, transform=val_transform, mode="test")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Verification of Dataset output
    sample_img, sample_label = train_dataset[0]
    assert sample_img.shape == (3, cfg.img_size, cfg.img_size), "Incorrect image shape"
    assert isinstance(sample_label, torch.Tensor), "Label should be a tensor"
    print("Dataset shapes verified.")

    # 4. Model Initialization
    print("\n[Model Initialization]")
    model = get_model(cfg, pretrained=True)  # Using pretrained for realistic test

    # Verify model output shape
    dummy_input = torch.randn(2, 3, cfg.img_size, cfg.img_size).to(cfg.device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    assert dummy_output.shape == (
        2,
        cfg.num_classes,
    ), f"Model output shape mismatch: {dummy_output.shape}"
    print("Model forward pass verified.")

    # 5. Loss Function & Optimizer
    print("\n[Loss & Optimizer]")
    # Move class weights to device for CrossEntropyLoss
    loss_fn = DistillationLoss(class_weights=class_weights.to(cfg.device))
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # 6. Training Loop (Fit)
    print("\n[Training Loop]")
    save_path = os.path.join(cfg.models_dir, "demo_model.pth")

    best_score = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=cfg.device,
        loss_fn=loss_fn,
        cfg=cfg,
        patience=1,  # Strict patience for demo
        save_path=save_path,
    )

    print(f"Training complete. Best Validation Score (AUC): {best_score:.4f}")
    assert os.path.exists(save_path), "Model checkpoint was not saved."

    # 7. Inference & Submission
    print("\n[Inference & Submission]")
    submission_path = os.path.join(cfg.working_dir, "demo_submission.csv")

    generate_submission(
        model=model,
        dataloader=test_loader,
        device=cfg.device,
        save_path=submission_path,
        target_cols=cfg.target_cols,
    )

    assert os.path.exists(submission_path), "Submission file not created."
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (
        len(df_test_demo),
        5,
    ), f"Submission shape mismatch: {df_sub.shape}"
    print(f"Submission generated at {submission_path}")
    print(df_sub.head())

    # 8. Distillation Mode Verification
    print("\n[Distillation Logic Verification]")
    # Simulate Teacher Logits for the training set
    # In a real scenario, these would come from a pre-trained teacher model
    # We create a dictionary mapping image_id to random logits
    teacher_logits_dict = {}
    for img_id in df_train_demo["image_id"]:
        teacher_logits_dict[img_id] = np.random.randn(cfg.num_classes).astype(
            np.float32
        )

    # Initialize Dataset in Distillation Mode
    distill_dataset = AppleDataset(
        df_train_demo,
        cfg,
        transform=train_transform,
        mode="distillation",
        teacher_logits=teacher_logits_dict,
    )

    # Fetch one item
    d_img, d_label, d_logits = distill_dataset[0]

    assert d_logits.shape == (
        cfg.num_classes,
    ), "Teacher logits shape mismatch in dataset"

    # Verify Loss function with teacher logits
    # Create a dummy batch
    d_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=4)
    b_img, b_label, b_t_logits = next(iter(d_loader))

    b_img = b_img.to(cfg.device)
    b_label = b_label.to(cfg.device)
    b_t_logits = b_t_logits.to(cfg.device)

    # Forward pass student
    student_logits = model(b_img)

    # Compute Distillation Loss
    d_loss = loss_fn(student_logits, b_label, teacher_logits=b_t_logits)

    assert not torch.isnan(d_loss), "Distillation loss is NaN"
    print(f"Distillation Loss computed successfully: {d_loss.item():.4f}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
