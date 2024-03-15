from gpt_language_model import *
import mmap
import random
import pickle


# Hyper-parameters
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 2e-5
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2


# # memory map for using small snippets of text from a single file of any size
# def get_random_chunk(split):
#     filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
#     with open(filename, 'rb') as f:
#         with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
#             # Determine the file size and a random position to start reading
#             file_size = len(mm)
#             start_pos = random.randint(0, (file_size) - block_size*batch_size)

#             # Seek to the random position and read the block of text
#             mm.seek(start_pos)
#             block = mm.read(block_size*batch_size-1)

#             # Decode the block to a string, ignoring any invalid byte sequences
#             decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
#             # Train and test splits
#             data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
#     return data


# def get_batch(split):
#     data = get_random_chunk(split)
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y


with open("dorothy_and_the_wizard_in_oz.txt", 'r', encoding='utf-8') as fo:
    text = fo.read()

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


model = GPTLanguageModel()
# print('loading model parameters...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loaded successfully!')
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


# saving model parameters
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')