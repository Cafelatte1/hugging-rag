import os
import pandas as pd
import numpy as np

class VectorRanker():
    def __init__(self, topN_chunks=10_000, n_retrievals=10, ranking_type="first_matching"):
        self.topN_chunks = topN_chunks
        self.n_retrievals = n_retrievals
        if ranking_type not in ['first_matching', 'equal_weighted', 'exponential_weighted']:
            raise ValueError("ranking_type must be in ['first_matching', 'equal_weighted', 'exponential_weighted']")
        else:
            self.ranking_type = ranking_type

    def __call__(self, corpus_container, scores, indicies):
        corpus_container["scores"] = -1.0
        # get averaged score on each documents
        # average score by equal weight and get Top N docs
        if self.ranking_type == "equal_weighted":
            # assign score
            corpus_container["scores"].iloc[indicies[0]] = scores[0]
            # get only topN chunks
            candidates = corpus_container.sort_values("scores", ascending=False).iloc[:self.topN_chunks]
            retrieval_docs = candidates.groupby("doc_id", sort=False, as_index=False)["scores"].apply(lambda x: x.mean()).sort_values("scores", ascending=False)
        # average score by exponential weight and get Top N docs
        elif self.ranking_type == "exponential_weighted":
            # assign score
            corpus_container["scores"].iloc[indicies[0]] = scores[0] * (np.logspace(start=0, stop=1, num=len(scores[0])) / 10.0)
            # get only topN chunks
            candidates = corpus_container.sort_values("scores", ascending=False).iloc[:self.topN_chunks]
            retrieval_docs = candidates.groupby("doc_id", sort=False, as_index=False)["scores"].apply(lambda x: x.mean()).sort_values("scores", ascending=False)
        # first matched N docs
        else:
            # assign score
            corpus_container["scores"].iloc[indicies[0]] = scores[0]
            # get only topN chunks
            candidates = corpus_container.sort_values("scores", ascending=False).iloc[:self.topN_chunks]
            retrieval_docs = candidates.drop_duplicates(subset="doc_id")
        retrieval_docs = retrieval_docs["doc_id"].iloc[:self.n_retrievals]
        return pd.concat([corpus_container[corpus_container["doc_id"] == doc_id].sort_values("scores", ascending=False) for doc_id in retrieval_docs.values], axis=0).reset_index(drop=True)
