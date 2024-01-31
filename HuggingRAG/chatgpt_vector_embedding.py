import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI

"""## Create Vector Embedding"""
    
class ChatGPTVectorEmbedding():
    def __init__(self, model_id, max_len):
        self.vectors = None
        self.store = None
        self.model_id = model_id
        self.max_len = max_len
        self.client = OpenAI()

    def get_vectorembedding(self, docs, norm=True):
        docs = [docs] if isinstance(docs, str) else docs        
        embed = torch.from_numpy(np.array([self.client.embeddings.create(input=[doc[:self.max_len]], model=self.model).data[0].embedding for doc in docs], dtype='float32'))
        embed = F.normalize(embed, p=2, dim=1).detach().cpu().numpy() if norm else embed.detach().cpu().numpy()
        return embed
