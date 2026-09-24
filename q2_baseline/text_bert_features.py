"""Shared aligned text_bert token-to-feature adapter."""

import numpy as np
import torch

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
FEATURE_RULE = "mean_last_four_hidden_layers_float32_zero_masked_tokens"


def load_encoder(device):
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device).eval()


@torch.no_grad()
def encode_text_bert(tokens, encoder, device, batch_size=16):
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or tokens.shape[1:] != (3, 50):
        raise ValueError(f"Expected (N,3,50), got {tokens.shape}")
    result = np.zeros((len(tokens), 50, 768), dtype=np.float32)
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start:start + batch_size]
        attention = np.asarray(batch[:, 1, :], dtype=np.int64)
        present = attention.any(axis=1)
        if not present.any():
            continue
        input_ids = torch.as_tensor(np.asarray(batch[present, 0, :], dtype=np.int64), device=device)
        attention_mask = torch.as_tensor(attention[present], device=device)
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True, return_dict=True)
        features = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)
        features *= attention_mask.unsqueeze(-1)
        result[np.flatnonzero(present) + start] = features.cpu().numpy().astype(np.float32)
    return result
