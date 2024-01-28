import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset
from utils import MeanPooling

"""## Create Vector Embedding"""

class VectorEmbedding():
    def __init__(self, model_id, max_length=512, device='cpu'):
        self.vectors = None
        self.store = None
        self.model_id = model_id
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer_params = {
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "return_token_type_ids": False,
            "return_tensors": "pt"
        }
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, x):
        tokens = self.tokenizer.batch_encode_plus(
            x, **self.tokenizer_params,
        )
        return tokens

    def get_vectorembedding(self, docs, batch_size=128, norm=True):
        embed = []
        pooler = MeanPooling()
        tokens = self.tokenize([docs] if isinstance(docs, str) else docs)
        dl = DataLoader(TensorDataset(tokens["input_ids"], tokens["attention_mask"]), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dl)):
                # input_ids
                batch[0] = batch[0].to(self.device)
                # attention_mask
                batch[1] = batch[1].to(self.device)
                output = self.model(**{"input_ids": batch[0], "attention_mask": batch[1]})
                embed.append(pooler(output.last_hidden_state, batch[1]))
                del batch, output
                torch.cuda.empty_cache()
                gc.collect()
        del pooler, tokens, dl
        torch.cuda.empty_cache()
        gc.collect()
        embed = torch.cat(embed, dim=0).to(torch.float32)
        return F.normalize(embed, p=2, dim=1).detach().cpu().numpy() if norm else embed.detach().cpu().numpy()

# # === EXAMPLE ===
# # create vector embedding class
# model_id = "microsoft/Multilingual-MiniLM-L12-H384"
# model_seq_len = 512
# vector_embedding = VectorEmbedding(model_id, max_length=model_seq_len)
# embedding = vector_embedding.get_vectorembedding(vector_data.get_chunks())