
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
# The code for training and evaluating a reward model using regression (all data).

# -----------------------------
# RewardModel with gating
# -----------------------------
class RewardModelGated(nn.Module):
    def __init__(self, emb_dim: int, gaze_dim: int = 0, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.emb_nll_mlp = nn.Sequential(
            nn.Linear(emb_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        if gaze_dim > 0:
            self.gaze_mlp = nn.Sequential(
                nn.Linear(gaze_dim, hidden_dim),
                nn.GELU()
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.gaze_mlp = None
            self.gate = None

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, emb, nll, gaze=None):
        x = torch.cat([emb, nll], dim=-1)
        h_text = self.emb_nll_mlp(x)
        if self.gaze_mlp is not None and gaze is not None and gaze.numel() > 0:
            h_gaze = self.gaze_mlp(gaze)
            gate_input = torch.cat([h_text, h_gaze], dim=-1)
            g = self.gate(gate_input)
            h = h_text + g * h_gaze
        else:
            h = h_text
        return self.out(h).squeeze(-1)


# -----------------------------
# Dataset for regression training
# -----------------------------
class RegressionSampleDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, float]], emb_dim, gaze_dim, device="cpu"):
        self.samples = samples
        self.device = device
        self.emb_dim = emb_dim
        self.gaze_dim = gaze_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats_np, label = self.samples[idx]
        emb = feats_np[:self.emb_dim]
        nll = feats_np[self.emb_dim:self.emb_dim+1]
        if self.gaze_dim > 0:
            gaze = feats_np[self.emb_dim+1:]
        else:
            gaze = np.zeros((0,), dtype=np.float32)

        emb = torch.tensor(emb, dtype=torch.float32, device=self.device)
        nll = torch.tensor(nll, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.float32, device=self.device)
        gaze = torch.tensor(gaze, dtype=torch.float32, device=self.device)
        return emb, nll, gaze, label


# -----------------------------
# Pairwise dataset for evaluation
# -----------------------------
class EvalPairDataset(Dataset):
    def __init__(self, df_pairs: pd.DataFrame, stats: Dict, gaze_cols: List[str], use_gaze: bool = True, device="cpu"):
        self.device = device
        self.use_gaze = use_gaze
        self.gaze_cols = gaze_cols if gaze_cols is not None else []
        self.stats = stats

        self.pairs = []
        for _, group in df_pairs.groupby("pair_id"):
            if len(group) != 2:
                continue
            row1, row2 = group.iloc[0], group.iloc[1]
            exp1, exp2 = float(row1["exp"]), float(row2["exp"])

            max_exp = max(exp1, exp2)
            if max_exp in [1, 2]:
                weight = 0.5
            elif max_exp in [1.25, 1.75]:
                weight = 0.25
            else:
                weight = 0.25
       
            if exp1 > exp2:
                pos, neg, exp_score = row1, row2, exp1
            elif exp2 > exp1:
                pos, neg, exp_score = row2, row1, exp2
            else:
                pos, neg, exp_score = row1, row2, 1.5

            self.pairs.append((pos, neg, exp_score, weight))

    def __len__(self):
        return len(self.pairs)

    def _normalize(self, x, col):
        mean, std = self.stats[col]["mean"], self.stats[col]["std"]
        return (x - mean) / (std + 1e-8)

    def _split_feats(self, row: pd.Series):
        emb = np.load(row["embedding_path"]).astype(np.float32)
        nll = np.array([self._normalize(row["nll"], "nll")], dtype=np.float32)
        if self.use_gaze and len(self.gaze_cols) > 0:
            gaze = np.array([self._normalize(row[col], col) for col in self.gaze_cols], dtype=np.float32)
        else:
            gaze = np.zeros((0,), dtype=np.float32)
        return emb, nll, gaze

    def __getitem__(self, idx):
        pos_row, neg_row, exp_score, weight = self.pairs[idx]
        pos_emb, pos_nll, pos_gaze = self._split_feats(pos_row)
        neg_emb, neg_nll, neg_gaze = self._split_feats(neg_row)

        return (
            torch.tensor(pos_emb, dtype=torch.float32, device=self.device),
            torch.tensor(pos_nll, dtype=torch.float32, device=self.device),
            torch.tensor(pos_gaze, dtype=torch.float32, device=self.device),
            torch.tensor(neg_emb, dtype=torch.float32, device=self.device),
            torch.tensor(neg_nll, dtype=torch.float32, device=self.device),
            torch.tensor(neg_gaze, dtype=torch.float32, device=self.device),
            torch.tensor(weight, dtype=torch.float32, device=self.device),
            float(exp_score),
        )


# -----------------------------
# Normalization stats
# -----------------------------
def compute_stats(df: pd.DataFrame, gaze_cols: List[str]) -> Dict:
    stats = {}
    stats["nll"] = {"mean": float(df["nll"].mean()), "std": float(df["nll"].std())}
    for col in (gaze_cols or []):
        stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}
    return stats


# -----------------------------
# Build regression samples
# -----------------------------
def build_regression_samples_from_pairs(df: pd.DataFrame, stats: Dict, gaze_cols: List[str], use_gaze: bool=True) -> List[Tuple[np.ndarray, float]]:
    samples = []
    for _, group in df.groupby("pair_id"):
        if len(group) != 2:
            continue
        row1, row2 = group.iloc[0], group.iloc[1]
        exp1, exp2 = float(row1["exp"]), float(row2["exp"])

        def feats_from_row(row):
            emb = np.load(row["embedding_path"]).astype(np.float32)
            nll = np.array([(row["nll"] - stats["nll"]["mean"]) / (stats["nll"]["std"] + 1e-8)], dtype=np.float32)
            if use_gaze and len(gaze_cols) > 0:
                gaze = np.array([(row[col] - stats[col]["mean"]) / (stats[col]["std"] + 1e-8) for col in gaze_cols], dtype=np.float32)
                feats = np.concatenate([emb, nll, gaze], axis=0)
            else:
                feats = np.concatenate([emb, nll], axis=0)
            return feats

        if exp1 > exp2:
            pos_row, neg_row, delta = row1, row2, exp1 - 1.5
        elif exp2 > exp1:
            pos_row, neg_row, delta = row2, row1, exp2 - 1.5
        else:
            pos_row, neg_row, delta = row1, row2, 0.0

        label_pos = 1.5 + delta
        label_neg = 1.5 - delta
        samples.append((feats_from_row(pos_row), float(label_pos)))
        samples.append((feats_from_row(neg_row), float(label_neg)))
    return samples


# -----------------------------
# Train one epoch
# -----------------------------
def train_regression_one_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for emb, nll, gaze, labels in dataloader:
        preds = model(emb, nll, gaze)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(dataloader)


# -----------------------------
# Evaluate regression 
# -----------------------------
def evaluate_regression(model, eval_pair_loader, device="cpu"):
    model.eval()
    weighted_correct, total_weight = 0.0, 0.0
    exp_correct, exp_total = 0, 0
    strong_correct, strong_total = 0, 0
    weak_correct, weak_total = 0, 0

    preds_list = []
    with torch.no_grad():
        for pos_emb, pos_nll, pos_gaze, neg_emb, neg_nll, neg_gaze, _, exp_score in eval_pair_loader:
            r1 = model(pos_emb, pos_nll, pos_gaze).cpu().numpy()
            r2 = model(neg_emb, neg_nll, neg_gaze).cpu().numpy()
            preds_list.extend(list(r1) + list(r2))
    preds_arr = np.array(preds_list)
    score_std = preds_arr.std()
    margin = 0.1 * score_std if score_std > 0 else 0.1

    with torch.no_grad():
        for pos_emb, pos_nll, pos_gaze, neg_emb, neg_nll, neg_gaze, weight, exp_score in eval_pair_loader:
            r1 = model(pos_emb, pos_nll, pos_gaze)
            r2 = model(neg_emb, neg_nll, neg_gaze)
            diff = r1 - r2
            tie_mask = (diff.abs() < margin)

            if exp_score == 1.5:
                correct = tie_mask.item()
            else:
                correct = diff.item() > 0

            exp_correct += int(correct)
            exp_total += 1
            if exp_score in [1, 2]:
                strong_correct += int(correct)
                strong_total += 1
            else:
                weak_correct += int(correct)
                weak_total += 1

            if exp_score != 1.5:
                weighted_correct += ((diff > 0).float().item() * weight.item())
                total_weight += weight.item()

    weighted_acc = weighted_correct / total_weight if total_weight > 0 else 0
    exp_acc = exp_correct / exp_total if exp_total > 0 else 0
    strong_acc = strong_correct / strong_total if strong_total > 0 else 0
    weak_acc = weak_correct / weak_total if weak_total > 0 else 0
    return weighted_acc, exp_acc, strong_acc, weak_acc


# -----------------------------
# Run K-Fold CV
# -----------------------------
def run_kfold_regression(
    data_path: str,
    gaze_cols: List[str],
    use_gaze: bool = True,
    k_folds: int = 5,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    lr: float = 5e-5,
    batch_size: int = 32,
    epochs: int = 25,
    device: str = "cpu",
    random_seed: int = 42
):
    df = pd.read_csv(data_path)
    pair_ids = np.array(df["pair_id"].unique())
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(pair_ids)):
        train_ids = pair_ids[train_idx]
        test_ids = pair_ids[test_idx]
        train_df = df[df["pair_id"].isin(train_ids)].reset_index(drop=True)
        test_df = df[df["pair_id"].isin(test_ids)].reset_index(drop=True)

        stats = compute_stats(train_df, gaze_cols if use_gaze else [])
        train_samples = build_regression_samples_from_pairs(train_df, stats, gaze_cols, use_gaze)
        np.random.shuffle(train_samples)

        emb_dim = np.load(train_df.iloc[0]["embedding_path"]).shape[0]
        gaze_dim = len(gaze_cols) if use_gaze else 0

        train_dataset = RegressionSampleDataset(train_samples, emb_dim, gaze_dim, device=device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        eval_pair_dataset = EvalPairDataset(test_df, stats, gaze_cols, use_gaze, device=device)
        eval_pair_loader = DataLoader(eval_pair_dataset, batch_size=1, shuffle=False)

        model = RewardModelGated(emb_dim, gaze_dim, hidden_dim, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        for epoch in range(epochs):
            _ = train_regression_one_epoch(model, train_loader, optimizer, device=device)

        w_acc, exp_acc, strong_acc, weak_acc = evaluate_regression(model, eval_pair_loader, device=device)
        print(f"Fold {fold+1}: WeightedAcc={w_acc:.3f}, ExpAcc={exp_acc:.3f}, StrongAcc={strong_acc:.3f}, WeakAcc={weak_acc:.3f}")
        fold_metrics.append({
            "fold": fold + 1,
            "w_acc": w_acc,
            "exp_acc": exp_acc,
            "strong_acc": strong_acc,
            "weak_acc": weak_acc
        })

    df_res = pd.DataFrame(fold_metrics)
    print("\n=== K-Fold Summary ===")
    print(df_res.mean(numeric_only=True))
    return df_res
