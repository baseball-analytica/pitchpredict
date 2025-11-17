# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

logger = logging.getLogger(__name__)

def collate_batch(
    pad_id: int,
    batch: list[tuple[torch.Tensor, torch.Tensor]],
):
    """
    Pads sequences to the longest in the batch.

    Args:
        batch: A list of tuples containing the input and output tensors.

    Returns:
        padded: (B, T)
        lengths: (B,)
        labels: (B,)
        perm_idx: indices used to sort by length desc (for packing)
        inv_perm_idx: inverse to restore original order if needed
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    
    # sort by length descending for pack_padded_sequence
    lengths_sorted, perm_idx = torch.sort(lengths, descending=True)
    perm_idx_list = perm_idx.tolist()
    seqs_sorted = [seqs[i] for i in perm_idx_list]
    labels_sorted = torch.stack([labels[i] for i in perm_idx_list])

    padded = pad_sequence(seqs_sorted, batch_first=True, padding_value=pad_id)

    # inverse permutation to restore original order
    inv_perm_idx = torch.empty_like(perm_idx)
    inv_perm_idx[perm_idx] = torch.arange(perm_idx.size(0))

    return padded, lengths_sorted, labels_sorted, perm_idx, inv_perm_idx


def accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Calculate the accuracy of the model.
    """
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """
    logger.debug("train_one_epoch called")

    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, lengths, y, _, _ in tqdm(loader, total=len(loader), desc="training"):
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        n += bsz

    logger.info("train_one_epoch completed successfully")
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the given data.
    """
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, lengths, y, _, _ in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        n += bsz

    return total_loss / n, total_acc / n


def train_model(
    model: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    model_path: str,
    pad_id: int,
) -> None:
    """
    Train the model on the given data.
    """
    logger.debug("train_model called")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_batch(pad_id, batch))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_batch(pad_id, batch))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = math.inf
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(f"epoch {epoch:02d} | train loss {train_loss:.4f} | train acc {train_acc:.3f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")

        # save the best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)

    logger.info(f"saved best model to {model_path}")
