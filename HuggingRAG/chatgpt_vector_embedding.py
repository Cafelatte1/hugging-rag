import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
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

    def get_vectorembedding(self, docs, batch_size=32, norm=True):
        docs = [docs] if isinstance(docs, str) else docs
        dl = DataLoader(docs, batch_size=batch_size, shuffle=False)
        embed = []
        for batch in tqdm(dl):
            output = self.client.embeddings.create(input=batch, model=self.model_id)
            embed.extend([np.array(i.embedding, dtype="float32") for i in output.data])
        embed = np.stack(embed, axis=0)
        embed = torch.from_numpy(np.array(embed))
        embed = F.normalize(embed, p=2, dim=1).detach().cpu().numpy() if norm else embed.detach().cpu().numpy()
        return embed
