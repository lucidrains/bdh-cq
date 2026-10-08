import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_enwik8 import SlidingWindow, sample


@pytest.mark.parametrize('num_tokens, num_windows', [(0, 0), (1, 0), (4, 0), (5, 1), (8, 1), (9, 2)])
def test_sliding_window_has_complete_targets(num_tokens, num_windows):
    dataset = SlidingWindow(torch.arange(num_tokens), seq_len = 4)

    assert len(dataset) == num_windows
    assert all(dataset[index].numel() == 5 for index in range(len(dataset)))

    for batch in DataLoader(dataset, batch_size = 2):
        assert batch.shape[-1] == 5


def test_sample_uses_prompt_logits_before_feeding_generated_tokens():
    class ContextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.context = []
            self.calls = []

        def forward(self, ids, memories = None, return_memory = False):
            self.calls.append(ids.tolist())
            self.context.extend(ids[0].tolist())
            logits = torch.full((*ids.shape, 256), -torch.inf)
            logits[..., len(self.context)] = 0.
            return logits, len(self.context)

    model = ContextModel()
    generated = sample(model, 'ab', length = 3, top_k = None)

    assert list(generated.encode('latin-1')) == [2, 3, 4]
    assert model.calls == [[[97, 98]], [[2]], [[3]]]
