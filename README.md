
## BDH-CQ (wip)

Implementation of <a href="https://arxiv.org/abs/2608.09888">BDH-CQ: In-Context Learning with Recurrent Latent Reasoning</a>, proposed by Pathway Research

## Install

```bash
$ pip install bdh-cq
```

## Usage

```python
import torch
from bdh_cq import BDH

model = BDH(
    dim = 512,
    num_tokens = 20_000
)

ids = torch.randint(0, 20_000, (2, 1024))

logits = model(ids) # (2, 1024, 20_000)
```

For recurrent latent reasoning, wrap the model and pass an interleaving of
token chunks and latent reasoning steps:

```python
from bdh_cq import BDH, BDHReasoningWrapper

model = BDH(
    dim = 512,
    num_tokens = 256
)

wrapper = BDHReasoningWrapper(model)

prompts = torch.randint(0, 256, (1, 64))
answers = torch.randint(0, 256, (1, 32))

# tensor stages are ingested, int stages are latent reasoning steps - any interleaving

loss, logits, memories = wrapper(prompts, 8, answers, return_loss = True, return_memory = True)

loss.backward()

# generate an answer

answer = wrapper.generate(prompts, 8, num_tokens = 32, stop_token = 0)
```

## Citations

```bibtex
@misc{engdahl2026bdhcq,
    title   = {BDH-CQ: In-Context Learning with Recurrent Latent Reasoning},
    author  = {Björn Engdahl and Adrian Kosowski and Jan Chorowski and Zuzanna Stamirowska and Przemysław Uznański and Junlin Jiang and Rohan Phadke and Remigiusz Kinas and Richard Zhong},
    year    = {2026},
    eprint  = {2608.09888},
    archivePrefix = {arXiv},
    primaryClass = {cs.NE},
    url     = {https://arxiv.org/abs/2608.09888}
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```
