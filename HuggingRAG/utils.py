import torch
from torch import nn

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # expand attention mask to embedding dimension for element-wise multiplication
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # element-wise multiplication & summation by embedding dim (if attention mask is zero, not pooled in final embedding vector)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # get the number of attention mask by each dims
        sum_mask = input_mask_expanded.sum(1)
        # clip to small number for prevent from div 0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # get final mean pooled embedding vector
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
