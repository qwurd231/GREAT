import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, kendalltau

# The code for training and evaluating a reward model using Bradley–Terry style pairwise loss (strong preference only).


# -----------------------------
# Dataset
# -----------------------------
class LikertPairDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, stats: dict, gaze_cols=None,
                 use_gaze=True, device="cpu"):
        self.device = device
        self.use_gaze = use_gaze
        self.gaze_cols = gaze_cols if gaze_cols is not None else []
        self.stats = stats

        self.samples = []
        for _, group in data_df.groupby("pair_id"):
            if len(group) != 2:
                continue
            row1, row2 = group.iloc[0], group.iloc[1]
            exp1, exp2 = float(row1["exp"]), float(row2["exp"])

            # strong preference / weak preference
            max_exp = max(exp1, exp2)
            if max_exp in [1, 2]:
                weight = 0.5
            elif max_exp in [1.25, 1.75]:
                continue
            else:
                continue  

            if exp1 < exp2:
                pos, neg = row2, row1
                exp_score = exp2
            else:
                pos, neg = row1, row2
                exp_score = exp1

            self.samples.append((pos, neg, weight, exp_score))

    def __len__(self):
        return len(self.samples)

    def _normalize(self, x, col):
        mean, std = self.stats[col]["mean"], self.stats[col]["std"]
        return (x - mean) / (std + 1e-8)

    def _to_tensor(self, sample):
        emb = np.load(sample["embedding_path"]).astype(np.float32)
        nll = np.array([self._normalize(sample["nll"], "nll")], dtype=np.float32)

        if self.use_gaze and len(self.gaze_cols) > 0:
            gaze = np.array([self._normalize(sample[col], col) for col in self.gaze_cols], dtype=np.float32)
            
        else:
            gaze = np.zeros((0,), dtype=np.float32)
    
        return (emb, nll, gaze)

    def __getitem__(self, idx):
        pos, neg, weight, exp_score = self.samples[idx]
        return (
            self._to_tensor(pos),
            self._to_tensor(neg),
            torch.tensor(weight, dtype=torch.float32, device=self.device),
            float(exp_score),
        )

    
class RewardModelGated(nn.Module):
    def __init__(self, emb_dim: int, gaze_dim: int = 0, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.emb_nll_mlp = nn.Sequential(
            nn.Linear(emb_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if gaze_dim > 0:
            self.gaze_mlp = nn.Sequential(
                nn.Linear(gaze_dim, hidden_dim),
                nn.GELU(),
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.gaze_mlp = None
            self.gate = None

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, emb, nll, gaze=None):
        x = torch.cat([emb, nll], dim=-1)
        h_text = self.emb_nll_mlp(x)

        if self.gaze_mlp is not None and gaze is not None:
            h_gaze = self.gaze_mlp(gaze)
            gate_input = torch.cat([h_text, h_gaze], dim=-1)
            g = self.gate(gate_input)
            h = h_text + g * h_gaze
        else:
            h = h_text

        return self.out(h).squeeze(-1)

# -----------------------------
# Loss
# -----------------------------
def weighted_pairwise_loss(r_pos, r_neg, weight):
    base_loss = -torch.log(torch.sigmoid(r_pos - r_neg) + 1e-12)
    return (base_loss * weight).mean()


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for pos_feats, neg_feats, weight, _ in dataloader:
        optimizer.zero_grad()
        emb_pos, nll_pos, gaze_pos = pos_feats
        emb_neg, nll_neg, gaze_neg = neg_feats

        r_pos = model(emb_pos, nll_pos, gaze_pos)
        r_neg = model(emb_neg, nll_neg, gaze_neg)
        loss = weighted_pairwise_loss(r_pos, r_neg, weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, compute_corr=False):
    device = next(model.parameters()).device
    model.eval()

    exp_correct, exp_total = 0, 0
    strong_correct, strong_total = 0, 0
    weak_correct, weak_total = 0, 0

    preds_all, labels_all = [], []

    with torch.no_grad():
        for pos_feats, neg_feats, _, exp_scores in dataloader:
         
            emb_pos, nll_pos, gaze_pos = pos_feats
            emb_neg, nll_neg, gaze_neg = neg_feats

            r_pos = model(emb_pos, nll_pos, gaze_pos)
            r_neg = model(emb_neg, nll_neg, gaze_neg)
        
            diff = (r_pos > r_neg).float()

            for i in range(len(exp_scores)):
                exp = exp_scores[i]
                correct = (diff[i].item() > 0)  
                exp_correct += int(correct)
                exp_total += 1
                if exp in [1.0, 2.0]:
                    strong_correct += int(correct)
                    strong_total += 1
                elif exp in [1.25, 1.75]:
                    weak_correct += int(correct)
                    weak_total += 1

            if compute_corr:
                preds_all.extend((r_pos - r_neg).cpu().numpy())
                labels_all.extend([1] * len(r_pos)) 

    results = {
        "exp_acc": exp_correct / exp_total if exp_total > 0 else 0,
        "strong_acc": strong_correct / strong_total if strong_total > 0 else 0,
        "weak_acc": weak_correct / weak_total if weak_total > 0 else 0,
    }

    if compute_corr and len(preds_all) > 0:
        results["spearman"] = spearmanr(preds_all, labels_all).correlation
        results["kendall"] = kendalltau(preds_all, labels_all).correlation

    return results

# -----------------------------
# normalization
# -----------------------------
def compute_stats(df: pd.DataFrame, gaze_cols):
    stats = {"nll": {"mean": df["nll"].mean(), "std": df["nll"].std()}}
    for col in gaze_cols:
        stats[col] = {"mean": df[col].mean(), "std": df[col].std()}
    return stats


def run_kfold_cv(
    data_path: str,
    gaze_cols=None,
    use_gaze=True,
    k_folds=5,
    hidden_dim=64,
    dropout=0.3,
    lr=5e-5,
    batch_size=32,
    epochs=40,
    device="cpu",
    compute_corr=False
):
    df = pd.read_csv(data_path)
    pair_ids = df["pair_id"].unique()
    np.random.seed(42)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pair_ids)):
        train_ids, test_ids = pair_ids[train_idx], pair_ids[test_idx]
        train_df, test_df = df[df["pair_id"].isin(train_ids)], df[df["pair_id"].isin(test_ids)]

        stats = compute_stats(train_df, gaze_cols if use_gaze else [])
        train_set = LikertPairDataset(train_df, stats, gaze_cols=gaze_cols, use_gaze=use_gaze, device=device)
        test_set = LikertPairDataset(test_df, stats, gaze_cols=gaze_cols, use_gaze=use_gaze, device=device)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        first_emb = np.load(train_df.iloc[0]["embedding_path"])
        emb_dim = len(first_emb)
        gaze_dim = len(gaze_cols) if use_gaze and gaze_cols is not None else 0

        model = RewardModelGated(emb_dim, gaze_dim=gaze_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
     
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer)

        results = evaluate(model, test_loader, compute_corr=compute_corr)
        results["fold"] = fold + 1
        fold_results.append(results)

    df_results = pd.DataFrame(fold_results)
    print("\n===== K-Fold Summary =====")
    print(df_results)
    print(df_results.mean(numeric_only=True))
    return df_results
