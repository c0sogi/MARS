import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold

from library.config import Config
from library.utils import seed_everything, compute_pearson
from library.dataset import load_and_prepare_data, PearsonDataset
from library.model import CustomModel
from library.engine import train_fn, valid_fn


def get_optimizer_params(model, encoder_lr, head_lr, weight_decay, llrd_decay):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # 1. Head Parameters (FC and Dropouts)
    # We identify head parameters as those belonging to 'fc' or 'dropouts' modules in CustomModel
    head_params = list(model.fc.named_parameters()) + list(
        model.dropouts.named_parameters()
    )

    optimizer_parameters.append(
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": head_lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": head_lr,
        }
    )

    # 2. Backbone Parameters
    # We iterate through the backbone to assign LLRD
    # DeBERTa-v3 structure: backbone.embeddings, backbone.encoder.layer.{i}

    backbone_named_params = list(model.backbone.named_parameters())
    num_layers = model.config.num_hidden_layers

    # Organize params by layer index
    # -1 for embeddings, 0 to num_layers-1 for encoder layers, num_layers for final layers (if any)
    layers = {i: [] for i in range(-1, num_layers + 1)}

    for n, p in backbone_named_params:
        if "embeddings" in n:
            layers[-1].append((n, p))
        elif "encoder.layer" in n:
            # Extract layer index
            # format: encoder.layer.X. ...
            try:
                parts = n.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])
                layers[layer_idx].append((n, p))
            except (ValueError, IndexError):
                # Fallback for unexpected naming, assign to top layer group
                layers[num_layers].append((n, p))
        else:
            # Other backbone params (e.g. rel_embeddings, final layernorm)
            layers[num_layers].append((n, p))

    # Assign LRs
    # Layer i LR = encoder_lr * (llrd_decay ^ (num_layers - 1 - i))
    # Top layer (i = num_layers - 1) gets encoder_lr
    # Embeddings (i = -1) gets lowest

    for layer_i, params in layers.items():
        if not params:
            continue

        if layer_i == -1:
            # Embeddings
            lr = encoder_lr * (llrd_decay**num_layers)
        elif layer_i == num_layers:
            # Final stuff / catch-all
            lr = encoder_lr
        else:
            # Encoder layers
            lr = encoder_lr * (llrd_decay ** (num_layers - 1 - layer_i))

        optimizer_parameters.append(
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            }
        )

    return optimizer_parameters


def run_training():
    """
    Orchestrates the entire training, validation, and submission pipeline.
    """
    seed_everything(Config.seed)

    # -------------------------------------------------------------------------
    # 1. Data Loading & Preparation
    # -------------------------------------------------------------------------
    print("Loading data...")
    # We load both train and val metadata to combine them for a proper CV split
    df_train_meta = load_and_prepare_data(Config.train_path)
    df_val_meta = load_and_prepare_data(Config.val_path)

    # Combine
    df_full = pd.concat([df_train_meta, df_val_meta]).reset_index(drop=True)

    if Config.debug:
        print("Debug mode: Sampling subset of data.")
        df_full = df_full.sample(n=1000, random_state=Config.seed).reset_index(
            drop=True
        )
        Config.epochs = 2  # Reduce epochs for debug

    print(f"Total training samples: {len(df_full)}")

    # -------------------------------------------------------------------------
    # 2. Cross-Validation Split (Stratified Group K-Fold)
    # -------------------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=Config.n_fold)

    # Create a new column for fold to easily select data later
    df_full["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(
            df_full, df_full[Config.stratify_col].astype(str), df_full[Config.group_col]
        )
    ):
        df_full.loc[val_idx, "fold"] = fold

    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Store validation scores

    for fold in range(Config.n_fold):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        train_df = df_full[df_full["fold"] != fold].reset_index(drop=True)
        valid_df = df_full[df_full["fold"] == fold].reset_index(drop=True)

        # Datasets & Loaders
        train_dataset = PearsonDataset(
            train_df, tokenizer, max_length=Config.max_length
        )
        valid_dataset = PearsonDataset(
            valid_df, tokenizer, max_length=Config.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model
        model = CustomModel(pretrained=True)
        model.to(Config.device)

        # Optimizer
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=Config.lr,
            head_lr=Config.head_lr,
            weight_decay=Config.weight_decay,
            llrd_decay=Config.llrd_decay,
        )
        optimizer = AdamW(
            optimizer_parameters, lr=Config.lr, eps=Config.eps, betas=Config.betas
        )

        # Scheduler
        num_train_steps = int(len(train_df) / Config.train_batch_size * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Loop Epochs
        best_pearson = -1.0
        best_model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")

        for epoch in range(Config.epochs):
            # Train
            train_loss = train_fn(
                train_loader, model, optimizer, scheduler, Config.device, epoch
            )

            # Validate
            val_preds, val_labels = valid_fn(valid_loader, model, Config.device)
            val_pearson = compute_pearson(val_labels, val_preds)

            print(f"Epoch {epoch+1} - Validation Pearson: {val_pearson}")

            # Save Best
            if val_pearson > best_pearson:
                print(
                    f"Score Improved ({best_pearson} -> {val_pearson}). Saving model..."
                )
                best_pearson = val_pearson
                torch.save(model.state_dict(), best_model_path)

        # Log best score
        print(f"Fold {fold} Best Pearson: {best_pearson}")

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            train_dataset,
            valid_dataset,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. Inference & Submission
    # -------------------------------------------------------------------------
    print(f"\n{'='*20} Inference on Test Set {'='*20}")

    df_test = load_and_prepare_data(Config.test_path)
    test_dataset = PearsonDataset(df_test, tokenizer, max_length=Config.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    final_predictions = np.zeros(len(df_test))

    for fold in range(Config.n_fold):
        print(f"Predicting with model fold {fold}...")
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")

        # Load model structure (no pretrained weights needed as we load state_dict)
        model = CustomModel(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)

        preds, _ = valid_fn(test_loader, model, Config.device)
        final_predictions += preds / Config.n_fold

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Create Submission File
    submission = pd.DataFrame({"id": df_test["id"], "score": final_predictions})

    # Clip scores to valid range [0, 1]
    submission["score"] = submission["score"].clip(0.0, 1.0)

    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(submission.head())
