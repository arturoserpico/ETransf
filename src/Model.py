import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        attn = F.softmax(scores, dim=-1)

        out = attn @ v

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.proj(out)
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            block_size
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
        return logits, loss


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model,
                    n_heads,
                    block_size
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(
            T,
            device=idx.device
        )

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
    
@torch.no_grad()
def generate(
    model,
    start_tokens,
    max_new_tokens,
):
    model.eval()

    idx = start_tokens

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -model.block_size:]

        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    return idx

def modelAndOpt(config):

    model = GPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
    ).to(config["device"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )

    return model, optimizer