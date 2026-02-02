import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything, compute_multilabel_auc
from library.dataset import get_datasets, NUM_CLASSES
from library.models import get_model
from library.sam import SAM
from library.trainer import Trainer


def run_demo():
    # 1. Setup and Reproducibility
    print("Step 1: Setting up environment and seeds...")
    seed_everything(42)

    # Define working directories for the demo
    demo_working_dir = "./working/demo_execution"
    checkpoint_dir = os.path.join(demo_working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Dataset Loading and Verification
    print("\nStep 2: Loading and processing datasets...")
    # We set load_cached_data=False to demonstrate processing from scratch using metadata and spectrograms
    # Note: dataset.py caches to ./working/idea_21 by default, but we will just use the returned objects.
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=False)

    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size: {len(val_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")

    # Verify Training Sample
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Train image should be a tensor"
    assert isinstance(label, torch.Tensor), "Train label should be a tensor"
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    assert label.shape == (NUM_CLASSES,), f"Unexpected label shape: {label.shape}"

    # Verify Test Sample (no label)
    test_img = test_dataset[0]
    assert isinstance(test_img, torch.Tensor), "Test image should be a tensor"
    assert test_img.shape == (
        3,
        224,
        224,
    ), f"Unexpected test image shape: {test_img.shape}"

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 3. Model Instantiation
    print("\nStep 3: Instantiating Model...")
    model_name = "resnet18"
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Model {model_name} loaded on {device}.")

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {NUM_CLASSES}), got {output.shape}"

    # 4. SAM Optimizer Demonstration
    print("\nStep 4: Configuring SAM Optimizer...")
    # SAM wraps a base optimizer (e.g., SGD or Adam)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9, rho=0.05)

    # Verify SAM logic with a toy example
    # Minimize f(x) = x^2, optimal x=0
    x = torch.tensor([10.0], requires_grad=True)
    sam_toy = SAM([x], torch.optim.SGD, lr=0.1, rho=0.05)

    # SAM requires a closure that calculates loss
    def closure():
        loss = x**2
        loss.backward()
        return loss

    # Perform one step
    loss_before = x.item() ** 2
    # Populate gradients for SAM
    loss_initial = x**2
    loss_initial.backward()
    sam_toy.step(closure)
    loss_after = x.item() ** 2

    print(f"  Toy SAM Optimization: Loss {loss_before:.4f} -> {loss_after:.4f}")
    assert (
        loss_after < loss_before
    ), "SAM optimizer failed to reduce loss in toy example"

    # 5. Training Loop Demonstration
    print("\nStep 5: Running Training Loop (Demo)...")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer, T_max=10
    )

    # Initialize Trainer
    # We use a very small number of epochs for demonstration speed
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        fold=0,
        epochs=2,  # Short run for demo
        patience=2,
        mixup_alpha=0.2,
        checkpoint_dir=checkpoint_dir,
    )

    # Run Training
    best_auc = trainer.fit()
    print(f"  Training completed. Best Validation AUC: {best_auc:.4f}")

    # Assert reasonable AUC (it might be low due to 2 epochs, but should be a float)
    assert isinstance(best_auc, float), "Best AUC should be a float"
    assert 0.0 <= best_auc <= 1.0, "AUC should be between 0 and 1"

    # 6. Prediction
    print("\nStep 6: Generating Predictions...")
    preds = trainer.predict(test_loader)

    print(f"  Prediction shape: {preds.shape}")
    assert preds.shape == (len(test_dataset), NUM_CLASSES), "Prediction shape mismatch"
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions should be probabilities between 0 and 1"

    # Generate submission file format
    submission_path = os.path.join(demo_working_dir, "submission", "submission.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Load test metadata to get rec_ids
    import pandas as pd

    test_meta = pd.read_csv("./metadata/test.csv")
    rec_ids = test_meta["rec_id"].values

    with open(submission_path, "w") as f:
        f.write("Id,Probability\n")
        for i, rec_id in enumerate(rec_ids):
            for class_idx in range(NUM_CLASSES):
                row_id = rec_id * 100 + class_idx
                prob = preds[i, class_idx]
                f.write(f"{row_id},{prob:.6f}\n")

    print(f"  Submission file generated at {submission_path}")

    # 7. Metric Utility Verification
    print("\nStep 7: Verifying Metric Utility...")
    # Create dummy targets and preds
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    y_pred = np.array(
        [[0.9, 0.1, 0.8], [0.2, 0.8, 0.3], [0.8, 0.7, 0.2], [0.1, 0.2, 0.9]]
    )

    # Calculate AUC
    # Class 0: True=[1,0,1,0], Pred=[0.9,0.2,0.8,0.1] -> Perfect separation -> AUC 1.0
    # Class 1: True=[0,1,1,0], Pred=[0.1,0.8,0.7,0.2] -> Perfect separation -> AUC 1.0
    # Class 2: True=[1,0,0,1], Pred=[0.8,0.3,0.2,0.9] -> Perfect separation -> AUC 1.0
    # Mean AUC should be 1.0
    auc_score = compute_multilabel_auc(y_true, y_pred)
    print(f"  Computed Dummy AUC: {auc_score:.4f}")
    assert np.isclose(
        auc_score, 1.0
    ), "Metric calculation incorrect for perfect predictions"

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
