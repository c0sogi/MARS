import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library modules
from library.config import Config
from library.utils import seed_everything, average_weights, calculate_metric
from library.data import DogDataset, get_transforms, get_classes
from library.model import DogClassifier
from library.engine import train_one_epoch, valid_one_epoch, create_greedy_soup


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. Configuration & Setup
    # Override Config for a fast demonstration
    Config.working_dir = "./working/demo_run"
    Config.batch_size = 8
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo
    Config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clean working directory
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seed
    seed_everything(Config.seed)
    print(f"Configuration set. Device: {Config.device}")

    # 2. Verify Utils
    print("\n[1/5] Verifying Utils...")

    # Test average_weights
    m1 = nn.Linear(1, 1)
    m2 = nn.Linear(1, 1)
    with torch.no_grad():
        m1.weight.fill_(1.0)
        m1.bias.fill_(1.0)
        m2.weight.fill_(3.0)
        m2.bias.fill_(3.0)

    avg_state = average_weights(m1.state_dict(), m2.state_dict(), alpha=0.5)

    # Expected average is 2.0
    assert torch.isclose(
        avg_state["weight"], torch.tensor([[2.0]])
    ).all(), "average_weights weight calculation incorrect"
    assert torch.isclose(
        avg_state["bias"], torch.tensor([2.0])
    ).all(), "average_weights bias calculation incorrect"

    # Test calculate_metric
    # Simple binary case masquerading as multiclass
    y_true_mock = np.array([0, 1])
    y_pred_mock = np.array([[0.9, 0.1], [0.1, 0.9]])
    loss = calculate_metric(y_true_mock, y_pred_mock)
    assert loss < 0.2, f"calculate_metric result too high for good preds: {loss}"
    print("Utils verified.")

    # 3. Verify Data
    print("\n[2/5] Verifying Data...")

    # Load metadata
    if not os.path.exists(Config.train_metadata_path):
        raise FileNotFoundError(f"Metadata not found at {Config.train_metadata_path}")

    df_full = pd.read_csv(Config.train_metadata_path)

    # Create a subset that contains exactly one example per class
    # This ensures valid_one_epoch -> log_loss receives all classes and doesn't crash due to shape mismatch
    df_subset = df_full.groupby("breed").head(1).reset_index(drop=True)
    print(f"Created subset with {len(df_subset)} samples (1 per class).")

    classes, class_to_idx = get_classes(df_full)
    assert len(classes) == 120, "Expected 120 classes"

    # Instantiate Dataset
    train_dataset = DogDataset(
        df_subset,
        transform=get_transforms(mode="train", image_size=224),
        mode="train",
        label_encoder=class_to_idx,
    )

    # Check single item
    img, label = train_dataset[0]
    assert img.shape == (3, 224, 224), f"Image shape mismatch: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"

    # Instantiate DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )

    # Check batch
    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape == (
        Config.batch_size,
        3,
        224,
        224,
    ), "Batch image shape mismatch"
    assert batch_labels.shape == (Config.batch_size,), "Batch label shape mismatch"
    print("Data verified.")

    # 4. Verify Model
    print("\n[3/5] Verifying Model...")
    # Initialize model (pretrained=False for speed/offline safety)
    model = DogClassifier(pretrained=False)
    model.to(Config.device)

    # Test Forward Pass
    with torch.no_grad():
        logits = model(batch_imgs.to(Config.device))
    assert logits.shape == (
        Config.batch_size,
        120,
    ), f"Logits shape mismatch: {logits.shape}"

    # Test Freeze/Unfreeze
    model.freeze_backbone()
    # Check first parameter of backbone
    assert not next(
        model.backbone.parameters()
    ).requires_grad, "Backbone should be frozen"
    # Check first parameter of head
    assert next(model.head.parameters()).requires_grad, "Head should be trainable"

    model.unfreeze_backbone()
    assert next(
        model.backbone.parameters()
    ).requires_grad, "Backbone should be trainable"
    print("Model verified.")

    # 5. Verify Engine (Training & Soup)
    print("\n[4/5] Verifying Engine...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # A. Train One Epoch
    print("Running training epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, Config.device, epoch=1)
    assert train_loss >= 0, "Training loss should be non-negative"

    # B. Valid One Epoch
    print("Running validation epoch...")
    # Use same loader for demo purposes
    val_loss, val_preds = valid_one_epoch(model, train_loader, Config.device)
    assert val_loss >= 0, "Validation loss should be non-negative"
    assert val_preds.shape == (len(df_subset), 120), "Prediction shape mismatch"

    # C. Create Greedy Soup
    print("Creating checkpoints for soup...")
    ckpt_paths = []

    # Save current model as checkpoint 1
    ckpt1_path = os.path.join(Config.working_dir, "model_epoch_1.pth")
    torch.save(model.state_dict(), ckpt1_path)
    ckpt_paths.append(ckpt1_path)

    # Modify model slightly and save as checkpoint 2 (simulate training progress)
    with torch.no_grad():
        for param in model.head.parameters():
            param.add_(0.01)

    ckpt2_path = os.path.join(Config.working_dir, "model_epoch_2.pth")
    torch.save(model.state_dict(), ckpt2_path)
    ckpt_paths.append(ckpt2_path)

    print("Constructing Greedy Soup...")
    soup_state = create_greedy_soup(model, train_loader, ckpt_paths, Config.device)
    assert soup_state is not None, "Soup generation returned None"
    assert isinstance(soup_state, dict), "Soup state should be a dict"

    # Save soup
    torch.save(soup_state, os.path.join(Config.working_dir, "best_soup.pth"))
    print("Engine verified.")

    print("\n[5/5] Demo Complete. All checks passed.")


if __name__ == "__main__":
    run_demo()
