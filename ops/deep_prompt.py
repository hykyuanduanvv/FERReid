"""Deep visual prompt insertion shared by the VPT and VICP models, and the WPA pair sampler."""
import torch


def sample_pair(labels, max_pairs=128):
    """Balanced positive / negative index pairs of a batch (for the WPA loss)."""
    labels = labels.cpu()
    i_idx, j_idx = torch.triu_indices(len(labels), len(labels), offset=1)
    pos = labels[i_idx] == labels[j_idx]
    positive = torch.stack((i_idx[pos], j_idx[pos]), dim=1)
    negative = torch.stack((i_idx[~pos], j_idx[~pos]), dim=1)
    n = min(max_pairs, len(positive), len(negative))
    pairs = torch.cat((positive[torch.randperm(len(positive))[:n]], negative[torch.randperm(len(negative))[:n]]))
    return pairs, torch.cat((torch.ones(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)))


def insert_deep_prompts(encoder, image_crops, prompts, num_layers):
    """Deep visual prompt tuning: prompts (n, L, V, D) are inserted after the CLS token of each of the last
    num_layers blocks (replacing the previous layer's prompt tokens). V is read from the tensor, so a prompt
    with extra domain tokens appended (adapters/active/prompt_tuning.py) works unchanged.
    Returns the raw CLS feature and the patch tokens."""
    prompts = prompts.reshape(prompts.size(0), num_layers, -1, prompts.size(-1))
    prompts = prompts[torch.randint(0, prompts.size(0), (image_crops.size(0),))]  # n = 1: shared prompt
    x = encoder.prepare_tokens_with_masks(image_crops, None)
    prompts = prompts.to(dtype=x.dtype)
    V = prompts.size(2)
    for blk in encoder.blocks[:-num_layers]:
        x = blk(x)
    for i, blk in enumerate(encoder.blocks[-num_layers:]):
        rest = x[:, 1:] if i == 0 else x[:, 1 + V:]
        x = blk(torch.cat([x[:, :1], prompts[:, i], rest], dim=1))
    x = encoder.norm(x)
    return x[:, 0], x[:, 1 + V:]
