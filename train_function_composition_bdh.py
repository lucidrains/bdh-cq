# one higher order BDH layer solves two-hop function composition f1(f2(x)), the order-1 BDH cannot
# run: python train_function_composition_bdh.py

import fire
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.optim import Adam

from einops import rearrange

from torch_einops_utils.shape import size

from bdh_cq import BDH, HigherOrderBDHLayer

def function_composition(seq_len, batch_size, num_classes = 10, composition_depth = 2, device = 'cpu'):
    inputs = torch.zeros((batch_size, seq_len, 2), dtype = torch.long, device = device)

    x_vals = torch.randint(0, num_classes, (batch_size,), device = device)
    funcs = torch.randint(0, num_classes, (batch_size, composition_depth, num_classes), device = device)
    targets = x_vals.clone()
    for step in range(composition_depth):
        targets = funcs[torch.arange(batch_size, device = device), step, targets]

    positions = torch.arange(min(composition_depth * num_classes, seq_len), device = device)
    step_idx, pos_idx = positions // num_classes, positions % num_classes

    inputs[:, :len(positions), 0] = funcs[torch.arange(batch_size, device = device)[:, None], step_idx, pos_idx]
    inputs[:, :len(positions), 1] = pos_idx
    inputs[:, -1, 0] = x_vals

    return inputs, targets

class Model(Module):
    def __init__(self, vocab_size, seq_len, attn_type = 'ho_bdh', layers = 1, dim = 128, heads = 4, dim_head = 32, rotary_dim = 32):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, dim)
        self.pos_enc = nn.Embedding(seq_len, dim)
        kwargs = dict(dim = dim, num_tokens = vocab_size, depth = layers, heads = heads, dim_qk_heads = heads * dim_head, rotary_dim = rotary_dim)

        self.net = BDH(block_cls = HigherOrderBDHLayer, **kwargs) if attn_type == 'ho_bdh' else BDH(**kwargs)

    def forward(self, x_one_hot):
        pos = torch.arange(size(x_one_hot, 'b [n] ...'), device = x_one_hot.device)
        return self.net(self.embedding(x_one_hot) + self.pos_enc(pos))

def train_model(attn_type, layers, epochs, batch_size, lr, device, num_classes = 10, composition_depth = 2, seed = 42):
    torch.manual_seed(seed)
    seq_len = composition_depth * num_classes + 1
    vocab_size = 3 + num_classes + num_classes

    model = Model(vocab_size = vocab_size, seq_len = seq_len, attn_type = attn_type, layers = layers).to(device)
    optimizer = Adam(model.parameters(), lr = lr)
    best_acc = 0.
    epoch_hit_1 = None

    for epoch in range(epochs):
        inputs, targets = function_composition(seq_len, batch_size = batch_size, num_classes = num_classes, composition_depth = composition_depth, device = device)

        b_in = rearrange(F.one_hot(inputs, num_classes = num_classes), 'b n f c -> b n (f c)').float()
        b_in = F.pad(b_in, (3, 0))
        b_in[:, :num_classes, 0] = b_in[:, num_classes:-1, 1] = b_in[:, -1, 2] = 1.
        optimizer.zero_grad()
        logits = model(b_in)
        loss = F.cross_entropy(logits[:, -1], targets)
        loss.backward()
        optimizer.step()

        acc = (logits[:, -1].argmax(dim = -1) == targets).float().mean().item()
        best_acc = max(best_acc, acc)

        if acc >= 1.:
            epoch_hit_1 = epoch + 1
            break
    return best_acc, epoch_hit_1

def main(epochs = 1500, batch_size = 256, lr = 1e-3, num_classes = 10, composition_depth = 2, seed = 42, bdh_layers = 4, device = 'mps'):
    if not torch.backends.mps.is_available() and device == 'mps':
        device = 'cpu'

    device = torch.device(device)
    train_fn = partial(train_model, epochs = epochs, batch_size = batch_size, lr = lr, num_classes = num_classes, composition_depth = composition_depth, seed = seed, device = device)

    results = [
        ("HigherOrderBDHLayer (1 layer)", *train_fn(attn_type = 'ho_bdh', layers = 1)),
        (f"BDH ({bdh_layers} layers)", *train_fn(attn_type = 'bdh', layers = bdh_layers))
    ]

    print("\nFinal Best Accuracies")

    for name, acc, epoch_hit in results:
        print(f"  {name}: {acc:.4f} (hit 1.0 at epoch {epoch_hit or 'never'})")

if __name__ == '__main__':
    fire.Fire(main)
