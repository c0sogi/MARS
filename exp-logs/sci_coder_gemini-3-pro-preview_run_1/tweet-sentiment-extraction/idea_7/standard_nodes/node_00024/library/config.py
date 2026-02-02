import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# Suppress warnings
warnings.filterwarnings("ignore")


class Config:
    # Hyperparameters
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    MODEL_PATH = "microsoft/deberta-v3-base"
    TOKENIZER_PATH = "microsoft/deberta-v3-base"
    SIGMA = 1.0  # For Gaussian smoothing
    SEED = 42
    NUM_WORKERS = 4

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_optimized"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata paths
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def process_data(df, tokenizer, max_len, is_test=False):
    n_samples = len(df)
    input_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask = np.zeros((n_samples, max_len), dtype=np.int32)
    offsets = np.zeros((n_samples, max_len, 2), dtype=np.int32)

    start_indices = np.zeros(n_samples, dtype=np.int32)
    end_indices = np.zeros(n_samples, dtype=np.int32)

    for idx, row in df.iterrows():
        text = " ".join(str(row["text"]).split())

        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True,
        )

        input_ids[idx] = encoded["input_ids"]
        attention_mask[idx] = encoded["attention_mask"]
        offsets[idx] = encoded["offset_mapping"]

        if not is_test:
            selected_text = " ".join(str(row["selected_text"]).split())
            start_char = text.find(selected_text)

            if start_char == -1:
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            tokens_offsets = encoded["offset_mapping"]
            start_token = 0
            end_token = 0
            found_start = False

            for i, (o_start, o_end) in enumerate(tokens_offsets):
                if o_start == 0 and o_end == 0:
                    continue
                if o_start <= start_char < o_end:
                    start_token = i
                    found_start = True
                    break

            if found_start:
                for i, (o_start, o_end) in enumerate(tokens_offsets):
                    if o_start == 0 and o_end == 0:
                        continue
                    if o_start < end_char <= o_end:
                        end_token = i
                        break

            if not found_start:
                start_token = 0
            if end_token == 0:
                # Fallback to last valid token
                end_token = np.sum(encoded["attention_mask"]) - 2

            start_indices[idx] = start_token
            end_indices[idx] = end_token

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offsets": offsets,
        "start_indices": start_indices,
        "end_indices": end_indices,
    }


def get_cached_data(df, tokenizer, config, cache_name="train", load_cached_data=True):
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    files = {
        "input_ids": os.path.join(config.CACHE_DIR, f"{cache_name}_input_ids.npy"),
        "attention_mask": os.path.join(
            config.CACHE_DIR, f"{cache_name}_attention_mask.npy"
        ),
        "offsets": os.path.join(config.CACHE_DIR, f"{cache_name}_offsets.npy"),
        "start_indices": os.path.join(
            config.CACHE_DIR, f"{cache_name}_start_indices.npy"
        ),
        "end_indices": os.path.join(config.CACHE_DIR, f"{cache_name}_end_indices.npy"),
    }

    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in files.values())
        if all_exist:
            print(f"Loading cached data for {cache_name}...")
            return {k: np.load(v) for k, v in files.items()}

    print(f"Processing data for {cache_name}...")
    is_test = "test" in cache_name
    data = process_data(df, tokenizer, config.MAX_LEN, is_test=is_test)

    for k, v in data.items():
        if k in files:
            np.save(files[k], v)

    return data


class TweetDataset(Dataset):
    def __init__(self, data, config, is_test=False):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.offsets = data["offsets"]
        self.is_test = is_test
        self.config = config

        if not is_test:
            self.start_indices = data["start_indices"]
            self.end_indices = data["end_indices"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        out = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
        }

        if not self.is_test:
            start_idx = self.start_indices[item]
            end_idx = self.end_indices[item]
            seq_len = self.config.MAX_LEN

            x = torch.arange(seq_len, dtype=torch.float)
            sigma = self.config.SIGMA

            start_target = torch.exp(-0.5 * ((x - start_idx) / sigma) ** 2)
            end_target = torch.exp(-0.5 * ((x - end_idx) / sigma) ** 2)

            start_target = start_target / (start_target.sum() + 1e-6)
            end_target = end_target / (end_target.sum() + 1e-6)

            out["start_targets"] = start_target
            out["end_targets"] = end_target

        return out


class SentimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.deberta = AutoModel.from_pretrained(config.MODEL_PATH)
        self.hidden_size = self.deberta.config.hidden_size

        self.layer_weights = nn.Parameter(torch.tensor([1 / 4] * 4))

        self.conv1 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1)
        self.conv3 = nn.Conv1d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv1d(
            self.hidden_size, self.hidden_size, kernel_size=5, padding=2
        )

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size * 3, 2)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        all_hidden_states = torch.stack(outputs.hidden_states[-4:])
        weights = F.softmax(self.layer_weights, dim=0).view(4, 1, 1, 1)
        pooled_output = (weights * all_hidden_states).sum(dim=0)

        x = pooled_output.permute(0, 2, 1)

        c1 = F.relu(self.conv1(x))
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))

        cat = torch.cat([c1, c3, c5], dim=1)
        cat = cat.permute(0, 2, 1)
        cat = self.dropout(cat)

        logits = self.fc(cat)

        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    start_loss = loss_fct(F.log_softmax(start_logits, dim=1), start_targets)
    end_loss = loss_fct(F.log_softmax(end_logits, dim=1), end_targets)
    return 0.5 * start_loss + 0.5 * end_loss


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, df_val):
    model.eval()
    start_preds = []
    end_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)
            start_preds.append(start_logits.cpu().numpy())
            end_preds.append(end_logits.cpu().numpy())

    start_preds = np.concatenate(start_preds)
    end_preds = np.concatenate(end_preds)

    predictions = []
    for i, row in df_val.iterrows():
        text = " ".join(str(row["text"]).split())
        offsets = dataloader.dataset.offsets[i]

        s_logits = start_preds[i]
        e_logits = end_preds[i]

        sum_logits = s_logits[:, None] + e_logits[None, :]
        mask = np.triu(np.ones_like(sum_logits))
        sum_logits = np.where(mask == 1, sum_logits, -np.inf)

        best_idx = np.argmax(sum_logits)
        best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

        if best_start < len(offsets) and best_end < len(offsets):
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            pred_text = text[start_char:end_char]
        else:
            pred_text = text
        predictions.append(pred_text)

    scores = [jaccard(p, a) for p, a in zip(predictions, df_val["selected_text"])]
    return np.mean(scores)


def run_pipeline():
    set_seed(Config.SEED)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    df_train = df_train[df_train["sentiment"] != "neutral"].reset_index(drop=True)
    df_val_model = df_val[df_val["sentiment"] != "neutral"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    train_data = get_cached_data(df_train, tokenizer, Config, "train_pos_neg")
    val_data = get_cached_data(df_val_model, tokenizer, Config, "val_pos_neg")

    train_dataset = TweetDataset(train_data, Config)
    val_dataset = TweetDataset(val_data, Config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentModel(Config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_jaccard = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.bin")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_jaccard = eval_epoch(model, val_loader, device, df_val_model)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), best_model_path)

    print("Starting Inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    df_test["selected_text"] = ""
    neutral_mask = df_test["sentiment"] == "neutral"
    df_test.loc[neutral_mask, "selected_text"] = df_test.loc[neutral_mask, "text"]

    non_neutral_mask = ~neutral_mask
    df_test_model = df_test[non_neutral_mask].reset_index(drop=True)

    if len(df_test_model) > 0:
        test_data = get_cached_data(df_test_model, tokenizer, Config, "test_pos_neg")
        test_dataset = TweetDataset(test_data, Config, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        start_preds = []
        end_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                s_logits, e_logits = model(input_ids, attention_mask)
                start_preds.append(s_logits.cpu().numpy())
                end_preds.append(e_logits.cpu().numpy())

        start_preds = np.concatenate(start_preds)
        end_preds = np.concatenate(end_preds)

        predictions = []
        for i, row in df_test_model.iterrows():
            text = " ".join(str(row["text"]).split())
            offsets = test_dataset.offsets[i]
            s_logits = start_preds[i]
            e_logits = end_preds[i]

            sum_logits = s_logits[:, None] + e_logits[None, :]
            mask = np.triu(np.ones_like(sum_logits))
            sum_logits = np.where(mask == 1, sum_logits, -np.inf)

            best_idx = np.argmax(sum_logits)
            best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

            if best_start < len(offsets) and best_end < len(offsets):
                start_char = offsets[best_start][0]
                end_char = offsets[best_end][1]
                pred_text = text[start_char:end_char]
            else:
                pred_text = text
            predictions.append(pred_text)

        df_test.loc[non_neutral_mask, "selected_text"] = predictions

    submission = df_test[["textID", "selected_text"]]
    submission.to_csv(Config.SUBMISSION_FILE, index=False, quoting=1)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


# Execute the pipeline
run_pipeline()
