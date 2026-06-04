import torch
import json
import Model

checkpoint = torch.load("data/model.pt", map_location='cuda')

config = json.load(open('data/config.json', 'r'))

model, optimizer = Model.modelAndOpt(config)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])

total_params = sum(p.numel() for p in model.parameters())

print(f"Total parameters: {total_params:,}")

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    """
    model: your GPT model
    idx: (B, T) starting token indices
    max_new_tokens: how many tokens to generate
    block_size: context window size
    """

    model.eval()

    for _ in range(max_new_tokens):

        # crop context if too long
        idx_cond = idx[:, -block_size:]

        # forward pass
        logits, _ = model(idx_cond)

        # take last time step
        logits = logits[:, -1, :] / temperature

        # optional: top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.tensor(float("-inf")).to(logits.device), logits)

        # convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # append to sequence
        idx = torch.cat((idx, next_token), dim=1)

    return idx


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


while(True):
    char = input("\ninput character: ")
    print("\n")

    # start with a single character (e.g. "T")
    start = torch.tensor([[stoi[char]]], device='cuda')

    out = generate(
        model=model,
        idx=start,
        max_new_tokens=300,
        block_size=64,
        temperature=0.8
    )

    # decode
    text = "".join(itos[i.item()] for i in out[0])
    print(text)