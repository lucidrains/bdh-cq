import torch

from bdh_cq import BDH, BDHReasoningWrapper


def test_greedy_generate_matches_teacher_forcing():
    # greedy decoding should agree with the training path teacher forced on its own output

    torch.manual_seed(0)

    model = BDH(dim = 64, num_tokens = 16, depth = 2, heads = 2, dim_qk_heads = 256, rotary_dim = 16)
    wrapper = BDHReasoningWrapper(model).eval()

    prompt = torch.randint(0, 16, (1, 24))

    with torch.no_grad():
        _, memories = wrapper(prompt, return_memory = True)
        tokens = wrapper.generate(3, memories = memories, num_tokens = 12, temperature = 0.)

        _, logits, _ = wrapper(3, torch.tensor([tokens]), memories = memories, return_loss = True, return_memory = True)

    assert logits[0, :-1].argmax(dim = -1).tolist() == tokens[1:]
