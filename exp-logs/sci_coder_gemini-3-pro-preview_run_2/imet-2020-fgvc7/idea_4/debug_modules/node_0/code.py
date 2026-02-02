import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings
import logging

# Suppress warnings and set logging level
warnings.filterwarnings("ignore")
logging.getLogger("ArtworkLabeller").setLevel(logging.ERROR)
logging.getLogger("dataset").setLevel(logging.ERROR)
logging.getLogger("modeling").setLevel(logging.ERROR)
logging.getLogger("inference").setLevel(logging.ERROR)

# Ensure library is in path
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders, ArtworkDataset
from library.losses import AsymmetricLoss, DistillationLoss
from library.modeling import (
    ArtworkClassifier,
    train_one_epoch,
    validate,
    generate_soft_labels,
    optimize_threshold,
)
from library.inference import predict_test_set, create_submission


def setup_demo_config():
    """
    Patches the Config class to run a fast, lightweight demo.
    """
    print("--- 1. Configuring Demo Environment ---")

    # Paths
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Checkpoints & Outputs
    Config.TEACHER_1_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_teacher.pth")
    Config.STUDENT_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_student.pth")
    Config.SOFT_LABELS_PATH = os.path.join(Config.WORKING_DIR, "demo_soft_labels.npy")
    Config.VAL_PREDS_PATH = os.path.join(Config.WORKING_DIR, "val_logits.npy")
    Config.VAL_TARGETS_PATH = os.path.join(Config.WORKING_DIR, "val_targets.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Runtime Settings
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.VAL_BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.IMAGE_SIZE = 224  # Smaller images for speed

    # Model Settings (Use lightweight ResNet18 instead of large Transformers/ConvNeXt)
    Config.TEACHER_1_MODEL_NAME = "resnet18"
    Config.TEACHER_2_MODEL_NAME = "resnet18"
    Config.STUDENT_MODEL_NAME = "resnet18"

    # Training Settings
    Config.TEACHER_EPOCHS = 1
    Config.STUDENT_EPOCHS = 1

    print(f"Config patched. Output dir: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}, Subset Size: {Config.DEBUG_SUBSET_SIZE}")


def verify_dataset_and_loader():
    print("\n--- 2. Verifying Dataset and DataLoader ---")

    dataloaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.VAL_BATCH_SIZE,
        debug=Config.DEBUG,
    )

    train_loader = dataloaders["train"]
    batch = next(iter(train_loader))

    # Check keys
    assert "image" in batch, "Batch missing 'image' key"
    assert "target" in batch, "Batch missing 'target' key"
    assert "id" in batch, "Batch missing 'id' key"

    # Check shapes
    images = batch["image"]
    targets = batch["target"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect target shape: {targets.shape}"

    print("Dataset verification successful.")
    return dataloaders


def verify_losses():
    print("\n--- 3. Verifying Loss Functions ---")

    batch_size = 4
    num_classes = Config.NUM_CLASSES

    # Dummy data
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    # 1. Asymmetric Loss
    asl = AsymmetricLoss()
    loss_val = asl(logits, targets)
    loss_val.backward()

    print(f"Asymmetric Loss Value: {loss_val.item():.4f}")
    assert loss_val.dim() == 0, "ASL should return a scalar"
    assert not torch.isnan(loss_val), "ASL returned NaN"

    # 2. Distillation Loss
    distill_loss = DistillationLoss(alpha=0.5)
    teacher_probs = torch.sigmoid(torch.randn(batch_size, num_classes))  # Soft targets

    # Reset grads
    logits.grad = None

    d_loss_val = distill_loss(logits, teacher_probs, targets)
    d_loss_val.backward()

    print(f"Distillation Loss Value: {d_loss_val.item():.4f}")
    assert d_loss_val.dim() == 0, "DistillationLoss should return a scalar"
    assert not torch.isnan(d_loss_val), "DistillationLoss returned NaN"

    print("Loss function verification successful.")


def demo_teacher_training(dataloaders):
    print("\n--- 4. Demonstrating Teacher Training (1 Epoch) ---")

    device = Config.DEVICE
    model = ArtworkClassifier(Config.TEACHER_1_MODEL_NAME, Config.NUM_CLASSES)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = AsymmetricLoss()

    # Train 1 epoch
    loss = train_one_epoch(
        model,
        dataloaders["train"],
        optimizer,
        None,  # No scheduler for demo
        criterion,
        device,
        epoch=1,
    )

    print(f"Teacher Train Loss: {loss:.4f}")
    assert loss > 0, "Training loss should be positive"

    # Save model
    torch.save(model.state_dict(), Config.TEACHER_1_CHECKPOINT)
    print(f"Teacher model saved to {Config.TEACHER_1_CHECKPOINT}")

    return model


def demo_soft_label_generation(teacher_model):
    print("\n--- 5. Demonstrating Soft Label Generation ---")

    # In a real run, we might use an ensemble. Here we use the single teacher trained above.
    teachers = [teacher_model]

    # Generate
    # Note: generate_soft_labels internally uses Config paths and settings
    soft_labels = generate_soft_labels(teachers, load_cached_data=False)

    print(f"Generated Soft Labels Shape: {soft_labels.shape}")

    # Verify shape matches debug subset size
    expected_len = Config.DEBUG_SUBSET_SIZE
    assert (
        len(soft_labels) == expected_len
    ), f"Soft labels count {len(soft_labels)} != Debug subset size {expected_len}"

    assert os.path.exists(Config.SOFT_LABELS_PATH), "Soft labels file not saved"
    print("Soft label generation verification successful.")


def demo_student_distillation():
    print("\n--- 6. Demonstrating Student Distillation ---")

    # Reload dataloaders, this time picking up the soft labels generated in step 5
    dataloaders = get_dataloaders(
        soft_labels_path=Config.SOFT_LABELS_PATH,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
    )

    # Verify soft targets are in batch
    batch = next(iter(dataloaders["train"]))
    assert "soft_target" in batch, "DataLoader failed to load soft targets"
    print("Soft targets successfully loaded into DataLoader.")

    # Initialize Student
    device = Config.DEVICE
    student = ArtworkClassifier(Config.STUDENT_MODEL_NAME, Config.NUM_CLASSES)
    student.to(device)

    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    criterion = DistillationLoss(alpha=0.5)

    # Train 1 epoch
    loss = train_one_epoch(
        student, dataloaders["train"], optimizer, None, criterion, device, epoch=1
    )

    print(f"Student Distillation Loss: {loss:.4f}")

    # Save student
    torch.save(student.state_dict(), Config.STUDENT_CHECKPOINT)
    return student, dataloaders


def demo_inference_pipeline(student_model, dataloaders):
    print("\n--- 7. Demonstrating Inference and Submission ---")

    device = Config.DEVICE
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 1. Validate & Optimize Threshold
    criterion = AsymmetricLoss()
    val_loss, val_logits, val_targets = validate(
        student_model, val_loader, criterion, device
    )

    best_thresh, best_f1 = optimize_threshold(val_logits, val_targets)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Optimal Threshold: {best_thresh:.4f}, Micro F1: {best_f1:.4f}")

    # 2. Predict on Test Set
    test_probs, test_ids = predict_test_set(student_model, test_loader, device)

    print(f"Test Predictions Shape: {test_probs.shape}")
    assert (
        len(test_ids) == test_probs.shape[0]
    ), "Mismatch between IDs and prediction rows"

    # 3. Create Submission
    create_submission(test_probs, test_ids, best_thresh, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head(3)}")
    assert (
        "id" in df_sub.columns and "attribute_ids" in df_sub.columns
    ), "Invalid submission format"

    print("Inference pipeline verification successful.")


if __name__ == "__main__":
    seed_everything(42)

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Data
        dataloaders = verify_dataset_and_loader()

        # 3. Losses
        verify_losses()

        # 4. Teacher Training
        teacher_model = demo_teacher_training(dataloaders)

        # 5. Soft Labels
        demo_soft_label_generation(teacher_model)

        # 6. Student Distillation
        student_model, dataloaders_with_soft = demo_student_distillation()

        # 7. Inference
        demo_inference_pipeline(student_model, dataloaders_with_soft)

        print("\n=== Demo Completed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e} !!!")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An Error Occurred: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
