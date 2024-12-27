import torch
import torch.nn as nn
import random
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt  
import time

batch_size = 264  # 3000
block_size = 8
n_embd = 32 
n_head = 4
n_layer = 4
dropout = 0.2
vocab_size = 202
device = 'cuda'

class Head(nn.Module): 
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
        

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
        
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.te = nn.Embedding(vocab_size, n_embd)
        self.pe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self,Xb, targets = None):
        B, T = Xb.shape
        te = self.te(Xb)
        pe = self.pe(torch.arange(T, device=device))
        x = te + pe
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            last_logits = logits[:, -1, :]  
            loss = F.cross_entropy(last_logits, targets.long())
        return logits, loss
    
    def generate(self, idx, max_new_tokens = 1):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -8:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
    
# Step 1: Define your search space    
batch_size = [200, 264, 420]  
block_size = [8, 16, 32, 128]
n_embd = [32, 64, 128]
n_head = [2, 4, 6, 8]
n_layer = [2, 4, 6, 8]
dropout = [0.2, 0.3, 0.5]
learning_rates = [0.001, 0.01, 0.1]

# Step 2: Set up the random search loop
best_accuracy = 0
best_params = {}

input_size = 32 

for _ in range(50):
    lr = random.choice(learning_rates)
    batch_size = random.choice(batch_size)
    hidden_size = random.choice(hidden_size)
    dropout_rate = random.choice(dropout)
    block_size = random.choice(dropout)
n_embd = [32, 64, 128]
n_head = [2, 4, 6, 8]
n_layer = [2, 4, 6, 8]
dropout = [0.2, 0.3, 0.5]