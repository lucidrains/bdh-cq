import math
import os
import zipfile
import urllib.request

import fire
import torch
import tqdm

from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from bdh_cq import BDH

# constants

DATA_DIR = './data'
ENWIK8_URL = 'http://mattmahoney.net/dc/enwik8.zip'

TRAIN_CHARS = 90_000_000
VALIDATE_CHARS = 10_000_000

SEED = 42

DIM = 384
DEPTH = 6
HEADS = 4
DIM_QK_HEADS = 2048

BATCH_SIZE = 8
SEQ_LEN = 512
STEPS = 5000
LR = 3e-4
WARMUP = 100

VALIDATE_EVERY = 500
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
TEMPERATURE = 0.9
TOP_K = 100

CHECKPOINT = './enwik8-bdh.pt'

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# data

def get_enwik8():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok = True)

    data_path = os.path.join(DATA_DIR, 'enwik8')
    zip_path = os.path.join(DATA_DIR, 'enwik8.zip')

    if not os.path.exists(data_path):
        print('downloading enwik8', flush = True)
        urllib.request.urlretrieve(ENWIK8_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip:
            zip.extractall(DATA_DIR)

    return open(data_path, 'rb').read()

def encode(data):
    return torch.frombuffer(bytearray(data), dtype = torch.uint8).long()

def decode(ids):
    return bytes(ids).decode('latin-1')

# dataset

class SlidingWindow(Dataset):
    def __init__(self, seq, seq_len):
        self.seq = seq
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (self.seq.numel() - 1) // self.seq_len)

    def __getitem__(self, ind):
        start = ind * self.seq_len
        return self.seq[start:(start + self.seq_len + 1)]

# model

def get_model(*, dim = DIM, depth = DEPTH, heads = HEADS, dim_qk_heads = DIM_QK_HEADS, attn_residual = True, attn_residual_tied = True, triadic = False, triadic_dim = 8):
    return BDH(
        dim = dim,
        num_tokens = 256,
        depth = depth,
        heads = heads,
        dim_qk_heads = dim_qk_heads,
        attn_residual = attn_residual,
        attn_residual_tied = attn_residual_tied,
        triadic = triadic,
        triadic_dim = triadic_dim
    )

# sampling through the recurrent memory

@torch.no_grad()
def sample(model, prompt, *, length = GENERATE_LENGTH, temperature = TEMPERATURE, top_k = TOP_K, device = None):
    model.eval()

    device = default(device, next(model.parameters()).device)

    prompt_ids = encode(prompt.encode('latin-1')).to(device)[None, :]

    # ingest the prompt, then generate one token at a time, carrying the memory

    memory = None
    logits, memory = model(prompt_ids, memories = memory, return_memory = True)

    sampled = []

    for index in range(length):
        if index > 0:
            logits, memory = model(prompt_ids, memories = memory, return_memory = True)

        logits = logits[0, -1] / temperature

        if exists(top_k):
            kth = logits.topk(top_k).values[-1]
            logits = logits.masked_fill(logits < kth, -torch.inf)

        token = torch.multinomial(logits.softmax(dim = -1), 1).item()
        sampled.append(token)

        prompt_ids = torch.tensor([[token]], device = device)

    model.train()

    return decode(sampled)

# train

def train(
    steps = STEPS,
    batch_size = BATCH_SIZE,
    seq_len = SEQ_LEN,
    lr = LR,
    validate_every = VALIDATE_EVERY,
    generate_every = GENERATE_EVERY,
    attn_residual = True,
    attn_residual_tied = True,
    triadic = False,
    triadic_dim = 8,
    wandb_log = True,
    checkpoint = CHECKPOINT,
    device = 'cuda'
):
    assert device == 'cpu' or (device.startswith('cuda') and torch.cuda.is_available()), f'device must be cpu or an available cuda, got {device}'

    torch.manual_seed(SEED)

    data = get_enwik8()
    train_ids, val_ids = encode(data[:TRAIN_CHARS]), encode(data[TRAIN_CHARS:(TRAIN_CHARS + VALIDATE_CHARS)])

    model = get_model(attn_residual = attn_residual, attn_residual_tied = attn_residual_tied, triadic = triadic, triadic_dim = triadic_dim).to(device)

    device_name = torch.cuda.get_device_name() if device.startswith('cuda') else device

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'BDH on {device_name}, {num_params:.1f}M params', flush = True)

    if wandb_log:
        try:
            import wandb
        except ImportError:
            print('wandb not installed, continuing without logging', flush = True)
            wandb_log = False

    if wandb_log:
        if not wandb.login(anonymous = 'never', force = False):
            print('no wandb api key found, logging offline (run `uv run wandb login` then `uv run wandb sync` to upload)', flush = True)
            os.environ['WANDB_MODE'] = 'offline'

        variant = 'attnres' if attn_residual else 'baseline'
        wandb.init(
            project = 'bdh-enwik8',
            name = f'{variant}-dim{DIM}-d{DEPTH}',
            config = dict(
                variant = variant,
                seed = SEED,
                steps = steps,
                batch_size = batch_size,
                seq_len = seq_len,
                lr = lr,
                dim = DIM,
                depth = DEPTH,
                heads = HEADS,
                dim_qk_heads = DIM_QK_HEADS,
                num_params = num_params
            )
        )

    opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.1, betas = (0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    )

    loader = DataLoader(SlidingWindow(train_ids, seq_len), batch_size = batch_size, shuffle = True, drop_last = True)

    prompt = decode(val_ids[:64])

    model.train()

    for step, batch in tqdm.tqdm(enumerate(loader), total = steps):
        if step >= steps:
            break

        batch = batch.to(device)

        loss = F.cross_entropy(rearrange(model(batch[:, :-1]), 'b n l -> b l n'), batch[:, 1:])
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)
        sched.step()

        if wandb_log and step % 100 == 0:
            wandb.log({'train_loss': loss.item()}, step = step)

        if step and step % validate_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(rearrange(model(val_ids[:(seq_len + 1)][None, :-1].to(device)), 'b n l -> b l n'), val_ids[1:(seq_len + 1)][None, :].to(device))
            model.train()
            bpb = val_loss.item() / math.log(2)
            print(f'\n[step {step}] val loss {val_loss.item():.3f}  bpb {bpb:.3f}', flush = True)

            if wandb_log:
                wandb.log({'val_loss': val_loss.item(), 'bpb': bpb}, step = step)

        if step and step % generate_every == 0:
            generated = sample(model, prompt)
            print(f'\n--- step {step} ---\nPROMPT: {prompt}\nGENERATE: {generated}\n', flush = True)

            if wandb_log:
                wandb.log({'generated': generated}, step = step)

    torch.save(model.state_dict(), checkpoint)
    print(f'\nsaved {checkpoint}', flush = True)

    generated = sample(model, prompt)
    print(f'\n--- final ---\n\nPROMPT: {prompt}\n\nGENERATE: {generated}\n', flush = True)

    if wandb_log:
        wandb.log({'generated': generated}, step = steps)
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(train)
