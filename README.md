
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

When using `block_cls = HigherOrderBDHLayer`, each output evaluates the two-hop
operator over its visible prefix. An earlier intermediate can read a later
position within that prefix, preserving the function-composition relationship;
positions beyond the current output cannot contribute. Full, chunked and
single-token calls use the same operation.

HigherOrder memory now retains historical intermediate keys, messages and masses
alongside the leaf statistics, so cache size grows with the number of stored
positions. Its computation also grows with context; it is not a constant-memory
linear recurrent layer. Existing model weights remain compatible, but old
two-tensor HigherOrder caches lack the required history and must be recreated by
re-ingesting the context. Layer calls return the complete updated cache; pass the
prior cache into the next call rather than merging independently computed caches.
Frozen memory writes retain the supplied cache.

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

```bibtex
@misc{knupp2026depthrecurrentattentionmixturesgiving,
    title   = {Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves},
    author  = {Jonas Knupp and Jan Hendrik Metzen and Jeremias Bohn and Georg Groh and Kristian Kersting},
    year    = {2026},
    eprint  = {2601.21582},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/2601.21582},
}
```

```bibtex
@inproceedings{chakrabarti2026poly,
    title   = {Poly-attention: a general scheme for higher-order self-attention},
    author  = {Chakrabarti, Sayak and Pitassi, Toniann and Alman, Josh},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year    = {2026}
}
```

```bibtex
@misc{sieberling2026triadic,
    title   = {Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling},
    author  = {Oliver Sieberling and Bharat Runwal and David Jin and Ryan Chin and Rameswar Panda and Yoon Kim},
    year    = {2026},
    eprint  = {2609.36529},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2609.36529}
}
```
