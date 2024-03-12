import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

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
# print(data[:100])

# divide the whole corpus into training data (0.8) and validating data (0.2)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

# blocks: multiple bigrams
# X is predictions and Y is targets
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('when input is', context, 'target is', target)

# CPU: process sequential tasks
# To train parallelly -> use GPU -> train multiple blocks parallely
# block size: the length of each sequence
# batch size: how many blocks can be trained at the same time
    
