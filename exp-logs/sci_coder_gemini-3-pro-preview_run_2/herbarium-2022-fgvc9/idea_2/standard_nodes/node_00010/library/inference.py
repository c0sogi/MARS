import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data_setup import get_test_dataloader, TaxonomyProcessor
from library.model import HierarchicalEfficientNet


def predict_test_set(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
    use_tta=False,
    load_cached_taxonomy=True,
):
    """
    Runs inference on the test set and saves the submission file.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (torch.device): Device to run inference on.
        use_tta (bool): Whether to use Test Time Augmentation (Horizontal Flip).
        load_cached_taxonomy (bool): Whether to load taxonomy counts from cache.
    """
    set_seed(Config.SEED)

    # 1. Get Taxonomy Counts for Model Initialization
    # We need num_families and num_genera to initialize the architecture
    processor = TaxonomyProcessor()
    _, counts = processor.process_taxonomy(load_cached_data=load_cached_taxonomy)

    num_families = counts["num_families"]
    num_genera = counts["num_genera"]
    num_species = counts["num_species"]

    # 2. Initialize Model
    model = HierarchicalEfficientNet(
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        pretrained=False,
    )

    # 3. Load Checkpoint
    checkpoint_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state dict key mismatch if model was saved with/without 'module.' prefix
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 4. Get Test DataLoader
    test_loader = get_test_dataloader(batch_size=batch_size, num_workers=num_workers)

    # 5. Inference Loop
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass (Original)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                species_logits = outputs["species"]

                if use_tta:
                    # TTA: Horizontal Flip
                    images_flipped = torch.flip(images, dims=[3])
                    outputs_flipped = model(images_flipped)
                    species_logits_flipped = outputs_flipped["species"]

                    # Average logits
                    species_logits = (species_logits + species_logits_flipped) / 2.0

            # Get predictions
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    # 6. Generate Submission
    submission_df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    # Sort by Id to ensure consistent ordering
    # Try to convert to int for sorting if possible
    try:
        submission_df["Id_int"] = submission_df["Id"].astype(int)
        submission_df = submission_df.sort_values("Id_int").drop(columns=["Id_int"])
    except ValueError:
        submission_df = submission_df.sort_values("Id")

    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
