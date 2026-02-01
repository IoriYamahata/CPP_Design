#!/usr/bin/env python
"""
Minimal CPP classifier: train and run predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


AMINO_ACID_DICT = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'PAD': 0,
}


def encode_sequences(seqs: pd.Series, max_len: int) -> torch.Tensor:
    out = []
    for seq in seqs:
        encoded = [AMINO_ACID_DICT.get(aa, 0) for aa in seq]
        padded = encoded + [AMINO_ACID_DICT['PAD']] * (max_len - len(encoded))
        out.append(padded)
    return torch.tensor(out, dtype=torch.long)


class CPPDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


class CPPClassifier(nn.Module):
    def __init__(self, num_amino_acids: int, d_model: int, num_classes: int, dropout_prob: float):
        super().__init__()
        self.embedding = nn.Embedding(num_amino_acids, d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.mean(x, dim=2)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for batch_data, in dataloader:
        batch_data = batch_data.to(device)
        outputs = model(batch_data).cpu().numpy()
        preds.append(outputs)
    return np.concatenate(preds, axis=0).reshape(-1)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    models_dir = repo_root / "models" / "predictor"
    models_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train CPP classifier and run predictions.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--filtered-csv", default=str(data_dir / "processed_data" / "filtered_sequence.csv"))
    parser.add_argument("--test-csv", default=str(data_dir / "processed_data" / "biomolecules-2162660_Supplementary Spreadsheets S1 Test Set.csv"))
    parser.add_argument("--lora-csv", default=str(data_dir / "generated_data" / "LoRA_9-18_residue.csv"))
    parser.add_argument("--oadm-csv", default=str(data_dir / "generated_data" / "oadm_9-18_residue.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPPClassifier(num_amino_acids=len(AMINO_ACID_DICT), d_model=16, num_classes=1, dropout_prob=args.dropout)
    model.to(device)

    best_path = models_dir / "best_model_state_dict.pt"

    if args.mode == "train":
        filtered_df = pd.read_csv(args.filtered_csv)
        test_df = pd.read_csv(args.test_csv)
        combined_df = pd.concat([filtered_df, test_df], ignore_index=True)

        max_len = combined_df['sequence'].apply(len).max()
        sequences = encode_sequences(combined_df['sequence'], max_len)
        labels = combined_df['class'].apply(lambda x: 1 if x == 'CPP' else 0).astype(int)
        labels = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)

        train_data, val_data, train_labels, val_labels = train_test_split(
            sequences, labels, test_size=1 - args.split, random_state=42
        )

        train_loader = DataLoader(CPPDataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(CPPDataset(val_data, val_labels), batch_size=args.batch_size, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for batch_data, batch_label in train_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_label in val_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item()

            avg_train = running_loss / max(1, len(train_loader))
            avg_val = val_loss / max(1, len(val_loader))
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), best_path)

            print(f"Epoch {epoch+1}/{args.epochs} - train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        print(f"Saved best model: {best_path}")
        return 0

    # predict mode
    if not best_path.exists():
        raise FileNotFoundError(f"Trained model not found: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    max_len = None
    for label, path in [("lora", args.lora_csv), ("oadm", args.oadm_csv)]:
        p = Path(path)
        if not p.exists():
            print(f"Skip prediction: {p} not found")
            continue
        df = pd.read_csv(p)
        if max_len is None:
            max_len = df['Sequence'].apply(len).max()
        seqs = encode_sequences(df['Sequence'], max_len)
        dl = DataLoader(seqs, batch_size=args.batch_size, shuffle=False)
        preds = predict(model, dl, device)
        out_path = models_dir / f"{label}_predictions.csv"
        pd.DataFrame({"Sequence": df['Sequence'], "Prediction": preds}).to_csv(out_path, index=False)
        print(f"Saved predictions: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
