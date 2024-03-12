import torch
import torch.nn as nn
import torch.nn.functional as F

# CPU: process sequential tasks
# To train parallelly -> use GPU -> train multiple blocks parallely
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
block_size = 8              # block size: the length of each sequence
batch_size = 4              # batch size: how many blocks can be trained at the same time
max_iters = 10000
learning_rate = 3e-4
# eval_interval = 2500
eval_iters = 250

# import the whole corpus
with open("dorothy_and_the_wizard_in_oz.txt", 'r', encoding='utf-8') as fo:
    text = fo.read()
chars = sorted(set(text))
vocabulary_size = len(chars)

# build an encoder and a decoder
# character level tokenizer
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# input corpus to torch as tensor instead of lists
data = torch.tensor(encode(text), dtype=torch.long)

# divide the whole corpus into training data (0.8) and validating data (0.2)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # embedding table: each row is a prediction distribution of the next token of current token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        return
    
    def forward(self, index, targets=None):
        # logits: a bunch of floating point numbers that are normalized
        # which is a probility distribution of what we want to predict
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            # B: Batch
            # T: Time
            # C: Channel (vocabulary size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # cross_entropy: a way to measure loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

model = BigramLanguageModel(vocabulary_size)
m = model.to(device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # ample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logts, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)  
    loss.backward()
    optimizer.step()
print(loss.item())

        