import torch
import torch.nn as nn
import random
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt  
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = ddp_rank == 0

torch.manual_seed(1337)
#random.seed(1337)

batch_size = 2640  # 3000
block_size = 8
n_embd = 32 
n_head = 4
n_layer = 4
dropout = 0.2
vocab_size = 2002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

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
        
# Generate batch
def DataLoader():
    a = random.randint(0, 1000) # 101
    b = random.randint(0, 1000) # 101
    target = a + b             # 201
    q = [a, 0, 0, b, 0,0,0,0]
    return q, target

torch.set_float32_matmul_precision('high')
model = TransformerModel(vocab_size, n_embd)
model = torch.compile(model)


if ddp:
    torch.cuda.set_device(ddp_local_rank)
    model = model.to(device)
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = optim.AdamW(model.parameters(), lr=3e-3)
losses = []

for step in range(100000):
    t0 = time.time()
    qs = []
    targets = []
    
    optimizer.zero_grad()
    
    for b in range(batch_size):
        q, target = DataLoader()
        qs.append(q)
        targets.append(target)
        
    qs = torch.tensor(qs, device = device)
    targets = torch.tensor(targets, device = device)  
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(qs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    losses.append(loss.item())
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}, dt: {dt:.2f}ms")
print(f"Final Loss: {loss.item()}") 



# -------------------------------------------------
model.eval()
all = 100

correct = 0
wrong = 0
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

for _ in range(all):
    q, t = DataLoader()
    context = torch.tensor([q], device = device)  # Shape (1, T)

 
    generated_sequence = raw_model.generate(context, max_new_tokens=1).tolist()
    generated_sequence = generated_sequence[0]
    
    #[a, 0, 0, b, 0,0,0,0]
    generated_sequence[1] = '+'
    
    del generated_sequence[2]
    del generated_sequence[3]
    del generated_sequence[3]
    del generated_sequence[3]
    generated_sequence[3] = '='
    if generated_sequence[-1] == t:
        correct += 1
        print("correct:")
        print(generated_sequence)
    else:
        wrong += 1
        print('wrong:')
        print(generated_sequence)

print("accuracy: ")
print(f"{correct} out of {all}")



plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('training_loss.png')


# loss: 1.58
# 0.97
#0.92
#1.05

# 0.28

# 1000 
# 2.326 :(
if ddp:
    destroy_process_group()
# torchrun --standalone --nproc_per_node=2 train.py