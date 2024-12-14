import torch
import torch.nn as nn
from torch.nn import functional as F
from nltk.tokenize import word_tokenize


class Head(nn.Module):
    def __init__(self, head_size, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = F.softmax(weights, dim=-1)

        out = weights @ v
        return out

    
class MultiHead(nn.Module):
    def __init__(self, num_head, head_size, d_model):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_model) for _ in range(num_head)])
        self.proj = nn.Linear(num_head * head_size, d_model)

    def forward(self, x):
        outputs = [head(x)[0] for head in self.heads]
        concat_output = torch.cat(outputs, dim=-1)
        output = self.proj(concat_output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size):
        super().__init__()
        head_size = d_model // num_heads
        self.attention = MultiHead(num_heads, head_size, d_model)
        self.mlp = FeedForward(d_model, hidden_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ln1 = self.ln1(x)
        attention_out = self.attention(x_ln1)
        x = x + attention_out
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, seq_length, num_heads, num_layers, n_hidden, n_output, device):
        super().__init__()
        self.seq_length = seq_length
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(seq_length, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, n_hidden) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))

        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    
class PositionalEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, seq_length, num_heads, num_layers, n_hidden, n_output, device):
        super().__init__()
        self.seq_length = seq_length
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = torch.zeros(seq_length, d_model, device=self.device)
        self.positional_encoding.requires_grad = False
        position = torch.arange(0, seq_length, device=self.device)
        position = position.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=self.device)
        self.positional_encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.positional_encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))


        self.blocks = nn.ModuleList([Block(d_model, num_heads, n_hidden) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)

        x = token_emb + self.positional_encoding[:T, :]
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x