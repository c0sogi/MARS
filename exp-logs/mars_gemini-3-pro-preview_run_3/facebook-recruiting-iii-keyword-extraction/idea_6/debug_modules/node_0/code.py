import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, data, model, engine


def run_demo():
    # =========================================================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # =========================================================================
    print(">>> Setting up demo environment...")

    # Define temporary directories for the demo
    demo_base_dir = "./working/demo_run"
    demo_input_dir = os.path.join(demo_base_dir, "input")
    demo_meta_dir = os.path.join(demo_base_dir, "metadata")
    demo_working_dir = os.path.join(demo_base_dir, "working")
    demo_submission_dir = os.path.join(demo_base_dir, "submission")

    for d in [demo_input_dir, demo_meta_dir, demo_working_dir, demo_submission_dir]:
        os.makedirs(d, exist_ok=True)

    # Override config paths to point to our demo directories
    config.INPUT_DIR = demo_input_dir
    config.METADATA_DIR = demo_meta_dir
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_submission_dir

    config.TRAIN_META_FILE = os.path.join(demo_meta_dir, "train.csv")
    config.VAL_META_FILE = os.path.join(demo_meta_dir, "val.csv")
    config.TEST_META_FILE = os.path.join(demo_meta_dir, "test.csv")

    config.TRAIN_RAW_FILE = os.path.join(demo_input_dir, "train.csv")
    config.TEST_RAW_FILE = os.path.join(demo_input_dir, "test.csv")

    # Override artifact paths
    config.TOKENIZER_PATH = os.path.join(demo_working_dir, "tokenizer.json")
    config.MLB_PATH = os.path.join(demo_working_dir, "mlb.joblib")
    config.TRAIN_FEATURES_PATH = os.path.join(demo_working_dir, "train_features.npy")
    config.TRAIN_LABELS_PATH = os.path.join(demo_working_dir, "train_labels.npy")
    config.VAL_FEATURES_PATH = os.path.join(demo_working_dir, "val_features.npy")
    config.VAL_LABELS_PATH = os.path.join(demo_working_dir, "val_labels.npy")
    config.TEST_FEATURES_PATH = os.path.join(demo_working_dir, "test_features.npy")
    config.TEST_IDS_PATH = os.path.join(demo_working_dir, "test_ids.npy")
    config.MODEL_PATH = os.path.join(demo_working_dir, "model.pth")
    config.SUBMISSION_FILE = os.path.join(demo_submission_dir, "submission.csv")

    # Override hyperparameters for speed
    config.VOCAB_SIZE = 100
    config.MAX_LEN = 20
    config.TOP_K_TAGS = 5
    config.EMBED_DIM = 16
    config.NUM_FILTERS = 8
    config.BATCH_SIZE = 4
    config.EPOCHS = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seeds
    utils.seed_everything(42)

    # =========================================================================
    # 2. CREATE MOCK DATA
    # =========================================================================
    print(">>> Creating mock data...")

    # Create Raw Data
    # 20 samples total
    ids = list(range(1, 21))
    titles = [f"Title for question {i}" for i in ids]
    bodies = [f"Body content for question {i} with some code and text." for i in ids]

    df_raw_train = pd.DataFrame({"Id": ids, "Title": titles, "Body": bodies})
    df_raw_train.to_csv(config.TRAIN_RAW_FILE, index=False)

    # Test raw file (same structure)
    test_ids = list(range(100, 105))
    test_titles = [f"Test Title {i}" for i in test_ids]
    test_bodies = [f"Test Body {i}" for i in test_ids]
    df_raw_test = pd.DataFrame(
        {"Id": test_ids, "Title": test_titles, "Body": test_bodies}
    )
    df_raw_test.to_csv(config.TEST_RAW_FILE, index=False)

    # Create Metadata
    # Train: IDs 1-10
    # Val: IDs 11-15
    # (IDs 16-20 unused in this split for simplicity)

    tags_pool = ["python", "java", "c++", "javascript", "html", "css"]

    def get_random_tags():
        return " ".join(
            np.random.choice(tags_pool, size=np.random.randint(1, 4), replace=False)
        )

    df_meta_train = pd.DataFrame(
        {
            "Id": ids[:10],
            "Tags": [get_random_tags() for _ in range(10)],
            "file_path": "train.csv",
        }
    )
    df_meta_train.to_csv(config.TRAIN_META_FILE, index=False)

    df_meta_val = pd.DataFrame(
        {
            "Id": ids[10:15],
            "Tags": [get_random_tags() for _ in range(5)],
            "file_path": "train.csv",
        }
    )
    df_meta_val.to_csv(config.VAL_META_FILE, index=False)

    df_meta_test = pd.DataFrame({"Id": test_ids, "file_path": "test.csv"})
    df_meta_test.to_csv(config.TEST_META_FILE, index=False)

    # =========================================================================
    # 3. VERIFY UTILS
    # =========================================================================
    print(">>> Verifying utils...")
    text = "Hello World! @2023 #coding"
    cleaned = utils.clean_text(text)
    assert cleaned == "hello world 2023 coding", f"Clean text failed: {cleaned}"

    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[0, 1, 0], [1, 0, 0]])
    f1 = utils.calculate_f1_score(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0, "F1 Score out of range"
    print("Utils verified.")

    # =========================================================================
    # 4. VERIFY DATA PIPELINE
    # =========================================================================
    print(">>> Verifying data pipeline...")

    # Force processing from scratch
    train_loader, val_loader, test_loader, tag_encoder = data.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Check Tokenizer existence
    assert os.path.exists(config.TOKENIZER_PATH), "Tokenizer not saved"

    # Check TagEncoder
    assert (
        len(tag_encoder.classes_) <= config.TOP_K_TAGS
    ), "TagEncoder classes exceeded Top K"

    # Check DataLoader output shapes
    for batch_x, batch_y in train_loader:
        assert batch_x.shape == (
            config.BATCH_SIZE,
            config.MAX_LEN,
        ), f"Train X shape mismatch: {batch_x.shape}"
        assert batch_y.shape == (
            config.BATCH_SIZE,
            len(tag_encoder.classes_),
        ), f"Train Y shape mismatch: {batch_y.shape}"
        break

    for batch_x, batch_ids in test_loader:
        assert batch_x.shape[1] == config.MAX_LEN, "Test X sequence length mismatch"
        assert batch_ids.shape[0] == batch_x.shape[0], "Test ID count mismatch"
        break

    print("Data pipeline verified.")

    # =========================================================================
    # 5. VERIFY MODEL
    # =========================================================================
    print(">>> Verifying model...")

    device = config.DEVICE
    net = model.WideDeepTextCNN(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_classes=len(tag_encoder.classes_),
        kernel_sizes=[3, 4],  # Small kernels for demo
        num_filters=config.NUM_FILTERS,
    ).to(device)

    # Create dummy input
    dummy_input = torch.randint(0, config.VOCAB_SIZE, (2, config.MAX_LEN)).to(device)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    assert output.shape == (
        2,
        len(tag_encoder.classes_),
    ), f"Model output shape mismatch: {output.shape}"
    print("Model verified.")

    # =========================================================================
    # 6. VERIFY ENGINE (TRAINING & INFERENCE)
    # =========================================================================
    print(">>> Verifying engine...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Train
    best_f1 = engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config.EPOCHS,
        patience=1,
        save_path=config.MODEL_PATH,
    )

    assert os.path.exists(config.MODEL_PATH), "Model file not saved after training"
    print(f"Training complete. Best F1: {best_f1:.4f}")

    # Threshold Search
    optimal_thresh = engine.find_optimal_threshold(net, val_loader, device)
    assert 0.0 < optimal_thresh < 1.0, "Optimal threshold out of expected range"

    # Submission Generation
    engine.generate_submission(
        model=net,
        test_loader=test_loader,
        tag_encoder=tag_encoder,
        device=device,
        output_path=config.SUBMISSION_FILE,
        threshold=optimal_thresh,
    )

    assert os.path.exists(config.SUBMISSION_FILE), "Submission file not created"

    # Verify submission content
    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"
    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission columns missing"

    print("Engine verified.")

    # =========================================================================
    # 7. CLEANUP
    # =========================================================================
    print(">>> Cleaning up...")
    shutil.rmtree(demo_base_dir)
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
