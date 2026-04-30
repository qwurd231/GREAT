import os
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# The code for training and evaluating a reward model using Bradley–Terry style pairwise loss (all data).


# -----------------------------
# Dataset
# -----------------------------
class LikertPairDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, stats: Dict, gaze_cols: List[str] = None,
                 use_gaze: bool = True, device="cpu"):
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

    
            max_exp = max(exp1, exp2)
            if max_exp in [1, 2]:
                weight = 0.5
            elif max_exp in [1.25, 1.75]:
                weight = 0.25
            else:  
                weight = 0.25

            if exp1 > exp2:
                pos, neg = row1, row2
                exp_score = exp1 
            elif exp2 > exp1:
                pos, neg = row2, row1
                exp_score = exp2
            else:
                pos, neg = row1, row2
                exp_score = 1.5 

            self.samples.append((pos, neg, exp_score, weight))


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
            gaze_tensor = torch.tensor(gaze, dtype=torch.float32, device=self.device)
        else:
            gaze = np.zeros((0,), dtype=np.float32)
            gaze_tensor = torch.tensor(gaze, dtype=torch.float32, device=self.device)

        return (
            torch.tensor(emb, dtype=torch.float32, device=self.device),
            torch.tensor(nll, dtype=torch.float32, device=self.device),
            gaze_tensor,
        )

    def __getitem__(self, idx):
        pos, neg, exp_score, weight = self.samples[idx]
        pos_emb, pos_nll, pos_gaze = self._to_tensor(pos)
        neg_emb, neg_nll, neg_gaze = self._to_tensor(neg)
        return (
            pos_emb,
            pos_nll,
            pos_gaze,
            neg_emb,
            neg_nll,
            neg_gaze,
            torch.tensor(weight, dtype=torch.float32, device=self.device),
            float(pos["mt"]),
            float(neg["mt"]),
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
def hybrid_loss(r_pos, r_neg, exp_scores, weight, alpha=0.5):
    """
    exp_scores == 1.5 -> tie: loss = alpha * |r_pos - r_neg|
    the other: pairwise logistic loss
    """
    is_tie = (exp_scores == 1.5).float()

    pairwise_loss = -torch.log(torch.sigmoid(r_pos - r_neg) + 1e-12)
    tie_loss = torch.abs(r_pos - r_neg)
    loss = (1 - is_tie) * pairwise_loss + is_tie * (alpha * tie_loss)
    return (loss * weight).mean()



def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for pos_emb, pos_nll, pos_gaze, neg_emb, neg_nll, neg_gaze, weight, _, _, exp_scores in dataloader:
        optimizer.zero_grad()
        r_pos = model(pos_emb, pos_nll, pos_gaze)
        r_neg = model(neg_emb, neg_nll, neg_gaze)
        loss = hybrid_loss(r_pos, r_neg, exp_scores.to(r_pos.device), weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)



def evaluate(model, dataloader, margin=0.1):
    device = next(model.parameters()).device
    model.eval()
    weighted_correct, total_weight = 0, 0
    exp_correct, exp_total = 0, 0
    strong_correct, strong_total = 0, 0
    weak_correct, weak_total = 0, 0
    preds_list = [] 

    with torch.no_grad():
        for pos_emb, pos_nll, pos_gaze, neg_emb, neg_nll, neg_gaze, weight, mt_pos, mt_neg, exp_scores in dataloader:
            r_pos = model(pos_emb, pos_nll, pos_gaze)
            r_neg = model(neg_emb, neg_nll, neg_gaze)

            exp_scores = torch.as_tensor(exp_scores, dtype=torch.float32, device=device)
            diff = r_pos - r_neg
            tie_mask = (diff.abs() < margin)

       
            weighted_correct += ((diff > 0).float() * (exp_scores != 1.5).float() * weight).sum().item()
            total_weight += ((exp_scores != 1.5).float() * weight).sum().item()

            # Exp-Acc
            for i in range(len(exp_scores)):
                exp = exp_scores[i].item()
                if exp == 1.5:  
                    correct = tie_mask[i].item()
                else:
                    correct = (diff[i].item() > 0) 
                exp_correct += int(correct)
                exp_total += 1

                # Strong vs Weak
                if exp in [1.0, 2.0]:
                    strong_correct += int(correct)
                    strong_total += 1
                else:
                    weak_correct += int(correct)
                    weak_total += 1
           

    weighted_acc = weighted_correct / total_weight if total_weight > 0 else 0
    exp_acc = exp_correct / exp_total if exp_total > 0 else 0
    strong_acc = strong_correct / strong_total if strong_total > 0 else 0
    weak_acc = weak_correct / weak_total if weak_total > 0 else 0

    return weighted_acc, exp_acc, strong_acc, weak_acc


# -----------------------------
# normalization
# -----------------------------
def compute_stats(df: pd.DataFrame, gaze_cols: List[str]) -> Dict:
    stats = {}
    stats["nll"] = {"mean": df["nll"].mean(), "std": df["nll"].std()}
    for col in gaze_cols:
        stats[col] = {"mean": df[col].mean(), "std": df[col].std()}
    return stats



def run_kfold_cv(
    data_path: str,
    gaze_cols: List[str],
    use_gaze: bool = True,
    k_folds: int = 5,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    lr: float = 5e-5,
    batch_size: int = 16,
    epochs: int = 25,
    device: str = "cpu",
):
    df = pd.read_csv(data_path)
    pair_ids = df["pair_id"].unique()
    np.random.seed(42)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(pair_ids)):
        train_ids, test_ids = pair_ids[train_idx], pair_ids[test_idx]
        train_df = df[df["pair_id"].isin(train_ids)]
        test_df = df[df["pair_id"].isin(test_ids)]

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

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)

        w_acc, exp_acc, strong_acc, weak_acc = evaluate(model, test_loader)
        print(f"Fold {fold+1}: WeightedAcc={w_acc:.3f}, ExpAcc={exp_acc:.3f}, "
        f"StrongAcc={strong_acc:.3f}, WeakAcc={weak_acc:.3f}")
        fold_results.append({"fold": fold + 1, "w_acc": w_acc, "exp_acc": exp_acc,
                             "strong_acc": strong_acc, "weak_acc": weak_acc})

    df_results = pd.DataFrame(fold_results)
    print("\n===== K-Fold Summary =====")
 
    print(df_results.mean(numeric_only=True))
    return df_results
