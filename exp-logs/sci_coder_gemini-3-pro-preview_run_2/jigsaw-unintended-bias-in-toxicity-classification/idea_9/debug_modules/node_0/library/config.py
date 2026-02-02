import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# =========================================================================================
# CONFIGURATION
# =========================================================================================
class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Model
    MODEL_NAME = "roberta-large"
    MAX_LEN = 512
    DROPOUT_SAMPLES = 5
    DROPOUT_RATE = 0.1

    # Training
    SEED = 42
    EPOCHS = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Loss & Bias Mitigation
    IDENTITY_SAMPLE_WEIGHT = 5.0
    AUX_LOSS_WEIGHT = 0.25

    # Identities
    IDENTITY_COLUMNS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]
    TARGET_COL = "target"


# =========================================================================================
# UTILS
# =========================================================================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================================================
# DATA PROCESSING
# =========================================================================================
def load_and_preprocess_data(config, load_cached_data=True):
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "train_ids": os.path.join(config.WORKING_DIR, "train_input_ids.npy"),
        "train_masks": os.path.join(config.WORKING_DIR, "train_masks.npy"),
        "train_targets": os.path.join(config.WORKING_DIR, "train_targets.npy"),
        "train_weights": os.path.join(config.WORKING_DIR, "train_weights.npy"),
        "train_aux": os.path.join(config.WORKING_DIR, "train_aux.npy"),
        "val_ids": os.path.join(config.WORKING_DIR, "val_input_ids.npy"),
        "val_masks": os.path.join(config.WORKING_DIR, "val_masks.npy"),
        "val_identities": os.path.join(config.WORKING_DIR, "val_identities.npy"),
        "test_ids": os.path.join(config.WORKING_DIR, "test_input_ids.npy"),
        "test_masks": os.path.join(config.WORKING_DIR, "test_masks.npy"),
        "test_df_ids": os.path.join(config.WORKING_DIR, "test_df_ids.npy"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading data from cache...")
        data = {k: np.load(v) for k, v in cache_files.items()}
        return data

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "validation.csv"))
    test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

    # Load Text
    print("Loading raw text files...")
    raw_train = pd.read_csv(
        os.path.join(config.INPUT_DIR, "train.csv"), usecols=["id", "comment_text"]
    )
    raw_test = pd.read_csv(
        os.path.join(config.INPUT_DIR, "test.csv"), usecols=["id", "comment_text"]
    )

    # Merge
    print("Merging text...")
    train_df = train_meta.merge(raw_train, on="id", how="left")
    val_df = val_meta.merge(raw_train, on="id", how="left")
    test_df = test_meta.merge(raw_test, on="id", how="left")

    # Fill NaNs
    train_df["comment_text"] = train_df["comment_text"].fillna("missing")
    val_df["comment_text"] = val_df["comment_text"].fillna("missing")
    test_df["comment_text"] = test_df["comment_text"].fillna("missing")

    # Tokenization
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def tokenize_df(df):
        encoded = tokenizer.batch_encode_plus(
            df["comment_text"].tolist(),
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    train_ids, train_masks = tokenize_df(train_df)
    val_ids, val_masks = tokenize_df(val_df)
    test_ids, test_masks = tokenize_df(test_df)

    # Targets & Weights
    print("Preparing targets and weights...")

    # Targets (Use raw fraction for soft labels or binary? Using raw as soft labels for BCE)
    train_targets = train_df[config.TARGET_COL].values.astype(np.float32)

    # Sample Weights: Weight = 5.0 if any identity is mentioned, else 1.0
    identity_sum = train_df[config.IDENTITY_COLUMNS].fillna(0).sum(axis=1)
    train_weights = np.where(
        identity_sum > 0, config.IDENTITY_SAMPLE_WEIGHT, 1.0
    ).astype(np.float32)

    # Aux Targets (Identities)
    train_aux = train_df[config.IDENTITY_COLUMNS].fillna(0).values.astype(np.float32)

    # Validation Data for Metrics (Identities + Target)
    val_identities = (
        val_df[config.IDENTITY_COLUMNS + [config.TARGET_COL]]
        .fillna(0)
        .values.astype(np.float32)
    )

    test_df_ids = test_df["id"].values

    # Save to cache
    np.save(cache_files["train_ids"], train_ids)
    np.save(cache_files["train_masks"], train_masks)
    np.save(cache_files["train_targets"], train_targets)
    np.save(cache_files["train_weights"], train_weights)
    np.save(cache_files["train_aux"], train_aux)
    np.save(cache_files["val_ids"], val_ids)
    np.save(cache_files["val_masks"], val_masks)
    np.save(cache_files["val_identities"], val_identities)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["test_masks"], test_masks)
    np.save(cache_files["test_df_ids"], test_df_ids)

    data = {
        "train_ids": train_ids,
        "train_masks": train_masks,
        "train_targets": train_targets,
        "train_weights": train_weights,
        "train_aux": train_aux,
        "val_ids": val_ids,
        "val_masks": val_masks,
        "val_identities": val_identities,
        "test_ids": test_ids,
        "test_masks": test_masks,
        "test_df_ids": test_df_ids,
    }
    return data


class ToxDataset(Dataset):
    def __init__(
        self, input_ids, attention_masks, targets=None, weights=None, aux_targets=None
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        self.weights = weights
        self.aux_targets = aux_targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float)
        if self.aux_targets is not None:
            item["aux_target"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)
        return item


# =========================================================================================
# MODEL
# =========================================================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, dim=1)
        c = torch.sum(last_hidden_state * w, dim=1)
        return c


class ToxicityModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(config.MODEL_NAME)
        self.pooler = AttentionPooling(self.roberta.config.hidden_size)

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(config.DROPOUT_RATE) for _ in range(config.DROPOUT_SAMPLES)]
        )

        # Heads
        self.toxicity_head = nn.Linear(self.roberta.config.hidden_size, 1)
        self.identity_head = nn.Linear(
            self.roberta.config.hidden_size, len(config.IDENTITY_COLUMNS)
        )

        self._init_weights(self.toxicity_head)
        self._init_weights(self.identity_head)
        self._init_weights(self.pooler)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.roberta.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state

        pooled_output = self.pooler(last_hidden_state, attention_mask)

        # Multi-Sample Dropout for Toxicity
        tox_logits = 0
        for dropout in self.dropouts:
            tox_logits += self.toxicity_head(dropout(pooled_output))
        tox_logits /= len(self.dropouts)

        # Identity Head (Auxiliary)
        ident_logits = self.identity_head(self.dropouts[0](pooled_output))

        return tox_logits, ident_logits


# =========================================================================================
# METRICS
# =========================================================================================
def calculate_bias_metrics(val_data_arr, y_pred, identity_cols):
    # val_data_arr columns: [identities..., target]
    val_df = pd.DataFrame(val_data_arr, columns=identity_cols + ["target"])
    val_df["prediction"] = y_pred
    val_df["label"] = (val_df["target"] >= 0.5).astype(int)

    overall_auc = roc_auc_score(val_df["label"], val_df["prediction"])

    bias_scores = []
    for col in identity_cols:
        subgroup_bool = val_df[col] >= 0.5
        if subgroup_bool.sum() == 0:
            continue

        # Subgroup AUC
        subgroup = val_df[subgroup_bool]
        sub_auc = (
            roc_auc_score(subgroup["label"], subgroup["prediction"])
            if len(subgroup["label"].unique()) > 1
            else 0.5
        )

        # BPSN
        bpsn_mask = ((val_df["label"] == 0) & subgroup_bool) | (
            (val_df["label"] == 1) & (~subgroup_bool)
        )
        bpsn = val_df[bpsn_mask]
        bpsn_auc = (
            roc_auc_score(bpsn["label"], bpsn["prediction"])
            if len(bpsn["label"].unique()) > 1
            else 0.5
        )

        # BNSP
        bnsp_mask = ((val_df["label"] == 1) & subgroup_bool) | (
            (val_df["label"] == 0) & (~subgroup_bool)
        )
        bnsp = val_df[bnsp_mask]
        bnsp_auc = (
            roc_auc_score(bnsp["label"], bnsp["prediction"])
            if len(bnsp["label"].unique()) > 1
            else 0.5
        )

        bias_scores.append([sub_auc, bpsn_auc, bnsp_auc])

    bias_scores = np.array(bias_scores)

    def power_mean(x, p=-5):
        return np.power(np.mean(np.power(x, p)), 1 / p)

    mp_subgroup = power_mean(bias_scores[:, 0])
    mp_bpsn = power_mean(bias_scores[:, 1])
    mp_bnsp = power_mean(bias_scores[:, 2])

    final_score = (
        0.25 * overall_auc + 0.25 * mp_subgroup + 0.25 * mp_bpsn + 0.25 * mp_bnsp
    )
    return final_score, overall_auc


# =========================================================================================
# TRAINING LOOP
# =========================================================================================
def train_and_predict(config):
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_and_preprocess_data(config)

    train_dataset = ToxDataset(
        data["train_ids"],
        data["train_masks"],
        data["train_targets"],
        data["train_weights"],
        data["train_aux"],
    )
    val_dataset = ToxDataset(data["val_ids"], data["val_masks"])
    test_dataset = ToxDataset(data["test_ids"], data["test_masks"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4
    )

    model = ToxicityModel(config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    num_train_steps = len(train_loader) * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_score = 0.0

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            weights = batch["weight"].to(device)
            aux_targets = batch["aux_target"].to(device)

            # Dynamic Trimming
            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            optimizer.zero_grad()
            tox_logits, ident_logits = model(input_ids, attention_mask)

            loss_tox = (criterion(tox_logits.view(-1), targets) * weights).mean()
            loss_aux = criterion(ident_logits, aux_targets).mean()

            total_loss = loss_tox + config.AUX_LOSS_WEIGHT * loss_aux
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()

        print(f"Epoch {epoch+1} Loss: {train_loss / len(train_loader)}")

        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                max_len = attention_mask.sum(dim=1).max().item()
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]

                tox_logits, _ = model(input_ids, attention_mask)
                preds = torch.sigmoid(tox_logits).view(-1).cpu().numpy()
                val_preds.extend(preds)

        score, overall_auc = calculate_bias_metrics(
            data["val_identities"], np.array(val_preds), config.IDENTITY_COLUMNS
        )
        print(f"Epoch {epoch+1} Validation Score: {score}")
        print(f"Epoch {epoch+1} Overall AUC: {overall_auc}")

        if score > best_score:
            best_score = score
            torch.save(
                model.state_dict(), os.path.join(config.WORKING_DIR, "best_model.pth")
            )

    # Inference
    model.load_state_dict(
        torch.load(os.path.join(config.WORKING_DIR, "best_model.pth"))
    )
    model.to(device)
    model.eval()

    print("Generating predictions...")
    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            tox_logits, _ = model(input_ids, attention_mask)
            preds = torch.sigmoid(tox_logits).view(-1).cpu().numpy()
            test_preds.extend(preds)

    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": data["test_df_ids"], "prediction": test_preds})
    submission.to_csv(
        os.path.join(config.SUBMISSION_DIR, "submission.csv"), index=False
    )
    print("Submission saved.")
