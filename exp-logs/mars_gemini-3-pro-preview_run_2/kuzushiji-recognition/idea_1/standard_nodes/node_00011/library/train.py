import os
import torch
import torch.optim as optim
from library.config import Config, seed_everything
from library.models import get_detection_model
from library.data import get_dataloaders


def train_detector(debug=False, epochs=Config.DET_EPOCHS):
    """
    Trains the Faster R-CNN Detection Model.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting Detection Training on {device}...")

    # 1. Data
    train_loader = get_dataloaders("train", debug=debug)
    # Note: Faster R-CNN in torchvision only computes loss in train mode.
    # Validation requires manual evaluation or custom loss computation.
    # For simplicity and speed, we will rely on training loss convergence and final validation.

    # Determine number of classes
    # train_loader.dataset is DetectionDataset
    num_classes = len(train_loader.dataset.label2id) + 1  # +1 for background
    print(f"Number of classes: {num_classes}")

    # 2. Model
    model = get_detection_model(num_classes=num_classes).to(device)

    # 3. Optimization
    # SGD is standard for Faster R-CNN
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=Config.DET_LR, momentum=0.9, weight_decay=0.0005)

    # Cite solution_lesson_node_00007: Delayed scheduler milestones (e.g. at epoch 8 and 11 for 12 epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 11], gamma=0.1
    )

    # 4. Training Loop
    save_path = os.path.join(Config.CACHE_DIR, "det_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), save_path)
    print("Detection training complete.")
    return save_path
