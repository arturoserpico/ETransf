import torch
from torch.utils.data import Dataset, DataLoader

import Model


# --------------------------
# Load dataset
# --------------------------

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")


# --------------------------
# Build vocabulary
# --------------------------

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {vocab_size}")


# --------------------------
# Encode entire dataset
# --------------------------

data = torch.tensor(
    [stoi[ch] for ch in text],
    dtype=torch.long
)

print(data.shape)


# --------------------------
# Train / validation split
# --------------------------

split_idx = int(0.5 * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]

print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))


# --------------------------
# Dataset class
# --------------------------

class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        x = self.data[idx : idx + self.block_size]

        y = self.data[idx + 1 : idx + self.block_size + 1]

        return x, y


# --------------------------
# Hyperparameters
# --------------------------

block_size = 128
batch_size = 32


# --------------------------
# Create datasets
# --------------------------

train_dataset = ShakespeareDataset(
    train_data,
    block_size
)

val_dataset = ShakespeareDataset(
    val_data,
    block_size
)


# --------------------------
# DataLoaders
# --------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)


# --------------------------
# Test batch
# --------------------------

x, y = next(iter(train_loader))

print("Input shape :", x.shape)
print("Target shape:", y.shape)

# Expected:
# torch.Size([32, 128])
# torch.Size([32, 128])

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "device": device,
    "vocab_size": vocab_size,
    "block_size": 128,
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 3,
}

model, optimizer = Model.modelAndOpt(config)

print(f'training data size: {len(train_loader)}')
print(f'selected device: {device}')

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0

    n = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, loss = model(x, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        print(f"\rtraining example: {n} / {len(train_loader)}", end="", flush=True)


        n += 1

    print()
    print(
        f"Epoch {epoch+1}: "
        f"{running_loss / len(train_loader):.4f}"
    )


torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}, "data/model.pt")

import json

with open("data/config.json", "w") as f:
    json.dump(config, f)