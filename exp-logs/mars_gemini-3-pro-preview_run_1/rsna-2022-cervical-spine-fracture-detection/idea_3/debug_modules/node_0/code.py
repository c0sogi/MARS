import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import glob

# Import library components
from library.config import Config
from library.utils import (
    process_dicom,
    calculate_weighted_loss,
    crop_image,
    apply_bone_window,
)
from library.models import SpineLocalizer, SliceEncoder, SequenceAggregator
from library.datasets import (
    SegmentationDataset,
    SliceClassificationDataset,
    FeatureSequenceDataset,
)
from library.losses import DiceLoss, EncoderLoss, WeightedLogLoss
from library.inference import predict_study


def run_demo():
    print("=== Starting Cervical Spine Fracture Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Monkey-patch Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5
    Config.LOCALIZER_EPOCHS = 1
    Config.ENCODER_EPOCHS = 1
    Config.SEQ_EPOCHS = 1
    Config.LOCALIZER_BATCH_SIZE = 2
    Config.ENCODER_BATCH_SIZE = 4
    Config.SEQ_BATCH_SIZE = 2

    # Initialize directories and seeds
    Config.setup()
    device = Config.DEVICE
    print(f"Device: {device}")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Loss Function
    y_true = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
    y_pred = np.array(
        [
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    loss = calculate_weighted_loss(y_true, y_pred)
    print(f"Calculated Weighted Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"

    # Test Image Processing (Find a valid DICOM)
    sample_study = test_df.iloc[0]
    sample_img_dir = os.path.join(Config.INPUT_DIR, sample_study["image_path"])
    dcm_files = glob.glob(os.path.join(sample_img_dir, "*.dcm"))

    if dcm_files:
        sample_dcm_path = dcm_files[0]
        print(f"Processing DICOM: {sample_dcm_path}")

        # Test process_dicom
        img = process_dicom(sample_dcm_path)
        print(
            f"Processed Image Shape: {img.shape}, Range: [{img.min():.2f}, {img.max():.2f}]"
        )
        assert img.shape == (512, 512), "Image shape mismatch"
        assert 0.0 <= img.min() and img.max() <= 1.0, "Image normalization failed"

        # Test crop_image
        crop = crop_image(img, center_yx=(256, 256), crop_size_hw=(128, 128))
        assert crop.shape == (128, 128), "Crop shape mismatch"
    else:
        print("No DICOM files found for utility test. Skipping image processing check.")

    # -------------------------------------------------------------------------
    # 3. Model Verification (Shapes & Forward Pass)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Models...")

    # A. Spine Localizer
    print("Testing SpineLocalizer...")
    localizer = SpineLocalizer(pretrained=False).to(device)
    dummy_input_loc = torch.randn(2, 1, 256, 256).to(device)  # Batch, Ch, H, W
    with torch.no_grad():
        out_loc = localizer(dummy_input_loc)
    print(f"Localizer Output Shape: {out_loc.shape}")
    assert out_loc.shape == (2, 1, 256, 256), "Localizer output shape mismatch"

    # B. Slice Encoder
    print("Testing SliceEncoder...")
    # Note: SliceEncoder expects 3 channels (2.5D stack)
    encoder = SliceEncoder(backbone_name=Config.ENCODER_BACKBONE, pretrained=False).to(
        device
    )
    dummy_input_enc = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out_enc = encoder(dummy_input_enc)
    print(f"Encoder Output Shape: {out_enc.shape}")
    # ResNet50 default features = 2048
    assert (
        out_enc.shape[0] == 2 and out_enc.shape[1] > 0
    ), "Encoder output shape mismatch"
    feature_dim = out_enc.shape[1]

    # C. Sequence Aggregator
    print("Testing SequenceAggregator...")
    aggregator = SequenceAggregator(input_dim=feature_dim).to(device)
    # Batch=2, SeqLen=10, FeatureDim
    dummy_input_agg = torch.randn(2, 10, feature_dim).to(device)
    with torch.no_grad():
        out_agg = aggregator(dummy_input_agg)
    print(f"Aggregator Output Shape: {out_agg.shape}")
    assert out_agg.shape == (
        2,
        8,
    ), "Aggregator output shape mismatch (should be 8 classes)"

    # -------------------------------------------------------------------------
    # 4. Dataset & Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Datasets & Training Steps...")

    # --- Stage 1: Localizer ---
    # Filter for a study that actually has segmentation file present
    seg_df = train_df[train_df["has_segmentation"] == True]

    # Verify file existence to pick a valid subset
    valid_indices = []
    for idx, row in seg_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["segmentation_path"])
        if os.path.exists(full_path):
            valid_indices.append(idx)
            if len(valid_indices) >= 2:
                break  # Just need a couple

    if valid_indices:
        print(f"Found {len(valid_indices)} valid segmentation samples for demo.")
        subset_seg_df = train_df.loc[valid_indices]

        # Instantiate Dataset
        # Note: This might take a moment to generate cache for these few samples
        seg_dataset = SegmentationDataset(
            metadata_df=subset_seg_df, load_cached_data=False
        )

        if len(seg_dataset) > 0:
            seg_loader = DataLoader(seg_dataset, batch_size=2)

            # Run 1 Training Step
            criterion_loc = DiceLoss()
            optimizer_loc = optim.Adam(localizer.parameters(), lr=1e-4)
            localizer.train()

            images, masks = next(iter(seg_loader))
            images, masks = images.to(device), masks.to(device)

            optimizer_loc.zero_grad()
            outputs = localizer(images)
            loss = criterion_loc(outputs, masks)
            loss.backward()
            optimizer_loc.step()

            print(f"Stage 1 (Localizer) Step Loss: {loss.item():.4f}")
        else:
            print("Segmentation dataset empty after processing.")
    else:
        print(
            "No valid segmentation files found in input/segmentations. Skipping Stage 1 data test."
        )

    # --- Stage 2: Encoder ---
    # Filter for studies with bounding boxes
    bbox_uids = set(pd.read_csv(Config.TRAIN_BBOXES_PATH)["StudyInstanceUID"])
    bbox_df = train_df[train_df["StudyInstanceUID"].isin(bbox_uids)].head(2)

    if not bbox_df.empty:
        print("Initializing SliceClassificationDataset...")
        # load_cached_data=False forces regeneration for this small subset
        enc_dataset = SliceClassificationDataset(
            metadata_df=bbox_df, load_cached_data=False
        )

        if len(enc_dataset) > 0:
            enc_loader = DataLoader(enc_dataset, batch_size=2)

            # Wrap encoder for training (add head)
            from library.train_stage2 import TrainableSliceEncoder

            trainable_encoder = TrainableSliceEncoder(
                Config.ENCODER_BACKBONE, pretrained=False
            ).to(device)

            criterion_enc = EncoderLoss()
            optimizer_enc = optim.Adam(trainable_encoder.parameters(), lr=1e-4)
            trainable_encoder.train()

            # Run 1 Training Step
            images, labels = next(iter(enc_loader))
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer_enc.zero_grad()
            logits = trainable_encoder(images)
            loss = criterion_enc(logits, labels)
            loss.backward()
            optimizer_enc.step()

            print(f"Stage 2 (Encoder) Step Loss: {loss.item():.4f}")
        else:
            print(
                "Encoder dataset empty (maybe no matching slices with bboxes in subset)."
            )
    else:
        print(
            "No training samples with bounding boxes found. Skipping Stage 2 data test."
        )

    # --- Stage 3: Aggregator ---
    print("Initializing FeatureSequenceDataset (Synthetic)...")
    # Generate synthetic features for the first 5 training samples
    subset_train_df = train_df.head(5)
    features_dict = {}
    for uid in subset_train_df["StudyInstanceUID"]:
        # Random sequence length between 50 and 100
        seq_len = np.random.randint(50, 100)
        features_dict[uid] = torch.randn(seq_len, feature_dim)

    agg_dataset = FeatureSequenceDataset(features_dict, subset_train_df)

    # Custom collate is needed for variable sequence lengths
    from library.train_stage3 import collate_fn

    agg_loader = DataLoader(agg_dataset, batch_size=2, collate_fn=collate_fn)

    criterion_agg = WeightedLogLoss()
    optimizer_agg = optim.Adam(aggregator.parameters(), lr=1e-4)
    aggregator.train()

    # Run 1 Training Step
    features, labels = next(iter(agg_loader))
    features, labels = features.to(device), labels.to(device)

    optimizer_agg.zero_grad()
    logits = aggregator(features)
    loss = criterion_agg(logits, labels)
    loss.backward()
    optimizer_agg.step()

    print(f"Stage 3 (Aggregator) Step Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference Simulation
    # -------------------------------------------------------------------------
    print("\n[5] Simulating Inference...")

    # Pick a test study
    test_uid = test_df.iloc[0]["StudyInstanceUID"]
    test_path = os.path.join(Config.INPUT_DIR, test_df.iloc[0]["image_path"])

    print(f"Running prediction on study: {test_uid}")

    # We use the models initialized earlier (untrained/random weights)
    # Ensure they are in eval mode
    localizer.eval()
    encoder.eval()  # Use the raw encoder, not the trainable wrapper
    aggregator.eval()

    models = (localizer, encoder, aggregator)

    try:
        probs = predict_study(test_uid, test_path, models, device)
        print(f"Prediction Probabilities: {probs}")
        assert len(probs) == 8, "Prediction output length mismatch"
        assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of range"
    except Exception as e:
        print(f"Inference failed (expected if no images found): {e}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
