import torch 
import torch.nn as nn
from torch.nn import functional as F


# Hyper-parameter
batch_size=64       # Number of sequence processed in parallel during one training step
block_size=256      # Max length of each input sequence.
max_iters=5000      # Total number of training steps
eval_interval=500   # Number of iteration between evaluation and validation
learning_rate=3e-6
device = "Cuda" if torch.cuda.is_available() else "CPU"
n_embd=384         # Dimensionality of each token
eval_iters=200      # Number of batched used when computing
n_head=6            # Number of Attention Head
n_layer=6           # Number of transformer layers
dropout=0.2

torch.manual_seed(1337)  # For Reproducibility

with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

# Unique charectors
chars=sorted(list(set(text)))
print(chars)
vocab_size= len(chars)
print(vocab_size)
# Characters to integer
stoi ={ch:i for i,ch in enumerate(chars)}
# Integer to characters
itos ={i:ch for i,ch in enumerate(chars)}

encode= lambda s: [stoi[c] for c in s]
decode= lambda l: ''.join([itos[i]for i in l])

# Train test split
data= torch.tensor(encode(text),dtype=torch.long)
n=int*(0.9*len(data))
train_data=data[:n]
val_data  =data[n:]

# Data loading
def get_batch(split):
    data= train_data if split=='train' else val_data
    ix=torch.randint( len(data)- block_size,(batch_size,))
    x= torch.stack([data[i:i+block_size]for i in ix])
    y=torch.stack(data[i+1:block_size+1]for i in ix)
    x,y =x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out ={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''One Head of Self Attention'''
    def __ini__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape

        k=self.key(x)
        q=self.query(x)

        wei = q @ k.transpose(-2,-1) * k.shpe[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        v=self.value(x)
        out=wei @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size)for _ in range(num_heads)])
        self.proj= nn.Linear(head_size * num_heads, n_embd)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        out= torch.cat([h(x) for h in self.heads],dim=-1) # concat
        out= self.dropout(self.proj(out))
        return out 
    
class FeedFoward(nn.Module):
    """Linear layer"""
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net()
    

class Block(nn.Module):
    """Transformer block """
    def __init__(self, n_embd,n_head):
        super().__init__()
        head_size=n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd= FeedFoward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head)for _ in range(n_layer)])
        self.ln_f =nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
           
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))