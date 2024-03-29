import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel

"""## Create Vector Embedding"""

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
    
class HuggingFaceVectorEmbedding():
    def __init__(self, model_id, tokenizer_max_length, device='cpu'):
        self.vectors = None
        self.store = None
        self.model_id = model_id
        self.device = torch.device("cuda" if device == "gpu" else device)
        self.max_length = tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def get_vector_embedding(self, docs, batch_size=32, norm=True):
        pooler = MeanPooling()
        tokenizer_params = {
            "max_length": self.max_length,
            "padding": "max_length",
            "truncation": True,
            "return_token_type_ids": False,
            "return_tensors": "pt"
        }
        tokens = self.tokenizer.batch_encode_plus([docs] if isinstance(docs, str) else docs, **tokenizer_params)
        dl = DataLoader(TensorDataset(tokens["input_ids"], tokens["attention_mask"]), batch_size=batch_size, shuffle=False)
        embed = []
        with torch.no_grad():
            for batch in dl:
                output = self.model(**{"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device)})
                embed.append(pooler(output.last_hidden_state, batch[1].to(self.device)).to(torch.float32))
                del batch, output
                torch.cuda.empty_cache()
                gc.collect()
        del pooler, tokens, dl
        embed = torch.cat(embed, dim=0)
        embed = F.normalize(embed, p=2, dim=1).detach().cpu().numpy() if norm else embed.detach().cpu().numpy()
        torch.cuda.empty_cache()
        gc.collect()
        return embed
