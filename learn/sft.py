"""
Supervised Fine-Tuning (SFT) following Chapter 7.
Uses PyTorch modules instead of hand-rolled helpers where possible.
"""

import json
import os
import re
import time
from functools import partial

import requests
import tiktoken
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import (
    download_and_load_gpt2,
    generate,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)


# ---------------------------------------------------------------------------
# Alpaca-style prompt format
# ---------------------------------------------------------------------------

def format_input(entry: dict) -> str:
    """Build the instruction + optional input prompt (no response)."""
    header = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    body = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return header + body


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """Pre-tokenises each (instruction, input, response) triple."""

    def __init__(self, data: list[dict], tokenizer):
        self.encoded = []
        for entry in data:
            prompt = format_input(entry)
            full_text = prompt + f"\n\n### Response:\n{entry['output']}"
            self.encoded.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


# ---------------------------------------------------------------------------
# Collate: pad_sequence + ignore_index masking
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[list[int]],
    pad_id: int = 50256,
    ignore_index: int = -100,
    max_len: int = 1024,
    device: str | torch.device = "cpu",
):
    """
    For each sequence:
      - append one EOS so the model learns to predict end-of-response
      - inputs  = seq[:-1]   (all but last token)
      - targets = seq[1:]    (shifted right by one)
      - mask every padding position in targets with ignore_index,
        but keep the first EOS so the model learns to stop

    Uses torch.nn.utils.rnn.pad_sequence for padding.
    """
    # Build input/target pairs and cap at max_len
    inputs, targets = [], []
    for ids in batch:
        seq = torch.tensor(ids + [pad_id])          # add EOS
        inp = seq[:-1][:max_len]
        tgt = seq[1:][:max_len]

        # Mask all padding in targets except the very first EOS
        eos_positions = (tgt == pad_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 1:
            tgt = tgt.clone()
            tgt[eos_positions[1:]] = ignore_index

        inputs.append(inp)
        targets.append(tgt)

    # pad_sequence pads to the longest sequence in the batch
    inputs_padded  = pad_sequence(inputs,  batch_first=True, padding_value=pad_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=ignore_index)

    return inputs_padded.to(device), targets_padded.to(device)


# ---------------------------------------------------------------------------
# Loss: cross_entropy with ignore_index handles padding natively
# ---------------------------------------------------------------------------

def calc_loss_batch(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)                              # (B, T, vocab)
    # ignore_index=-100 skips padded positions automatically
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float("nan")
    n = min(num_batches, len(loader)) if num_batches is not None else len(loader)
    total = sum(
        calc_loss_batch(x, y, model, device).item()
        for i, (x, y) in enumerate(loader) if i < n
    )
    return total / n


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int = 5,
    eval_iter: int = 5,
    start_context: str = "",
    tokenizer=None,
):
    train_losses, val_losses, tokens_seen_log = [], [], []
    tokens_seen, step = 0, -1
    context_size = model.pos_emb.weight.shape[0]

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += x.numel()
            step += 1

            if step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    tr = calc_loss_loader(train_loader, model, device, eval_iter)
                    vl = calc_loss_loader(val_loader,   model, device, eval_iter)
                model.train()
                train_losses.append(tr)
                val_losses.append(vl)
                tokens_seen_log.append(tokens_seen)
                print(f"Ep {epoch+1:02d} step {step:06d} | train {tr:.4f}  val {vl:.4f}")

        # Sample after each epoch
        if tokenizer and start_context:
            model.eval()
            ids = generate(
                model=model,
                idx=text_to_token_ids(start_context, tokenizer).to(device),
                max_new_tokens=50,
                context_size=context_size,
                eos_id=50256,
            )
            print("Sample:", token_ids_to_text(ids, tokenizer).replace("\n", " "))
            model.train()

    return train_losses, val_losses, tokens_seen_log


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str, url: str) -> list[dict]:
    if not os.path.exists(path):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = load_data(
        "instruction-data.json",
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
        "ch07/01_main-chapter-code/instruction-data.json",
    )

    n_train = int(len(data) * 0.85)
    n_test  = int(len(data) * 0.10)
    train_data = data[:n_train]
    test_data  = data[n_train : n_train + n_test]
    val_data   = data[n_train + n_test:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    collate   = partial(collate_fn, device=device, max_len=1024)

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=8, collate_fn=collate, shuffle=True,  drop_last=True,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=8, collate_fn=collate, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer),
        batch_size=8, collate_fn=collate, shuffle=False, drop_last=False,
    )

    # ------------------------------------------------------------------
    # Pretrained GPT-2 model
    # ------------------------------------------------------------------
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()
    print(f"Loaded: {CHOOSE_MODEL}")

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    print("\nPre-training losses:")
    with torch.no_grad():
        print(f"  train: {calc_loss_loader(train_loader, model, device, 5):.4f}")
        print(f"  val:   {calc_loss_loader(val_loader,   model, device, 5):.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 2

    t0 = time.time()
    train_losses, val_losses, _ = train(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )
    print(f"\nTraining done in {(time.time()-t0)/60:.2f} min")
    print(f"Final — train: {train_losses[-1]:.4f}  val: {val_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # Generate responses for test set
    # ------------------------------------------------------------------
    model.eval()
    for i, entry in enumerate(test_data):
        prompt = format_input(entry)
        ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        full   = token_ids_to_text(ids, tokenizer)
        test_data[i]["model_response"] = full[len(prompt):].replace("### Response:", "").strip()

    out_path = "instruction-data-with-response.json"
    with open(out_path, "w") as f:
        json.dump(test_data, f, indent=4)
    print(f"Responses saved to {out_path}")

    model_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    main()
