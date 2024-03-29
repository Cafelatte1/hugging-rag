import faiss

class FaissVectorStore():
    def __init__(self, vector_data, ranker, similarity_algorithm="dot_product", k=1_000_000):
        self.vector_data = vector_data
        self.store = None
        self.ranker = ranker
        if similarity_algorithm not in ["dot_product"]:
            ValueError(f"{self.similarity_algorithm} is not supported.")
        else:
            self.similarity_algorithm = similarity_algorithm
        self.k = k
        self.max_gpu_k = 2048

    def get_vector_store(self, embedding, use_gpu=False):
        if self.similarity_algorithm == "dot_product":
            self.store = faiss.IndexFlatIP(embedding.shape[-1])
        else:
            ValueError(f"{self.similarity_algorithm} is not supported.")
        if use_gpu:
            self.store = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.store)
        self.store.add(embedding)

    def search(self, embedding, use_gpu=False, excluding_zero_score=True):
        # transform (E) -> (1, E)
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        scores, indicies = self.store.search(embedding, k=min(self.max_gpu_k, self.k) if use_gpu else self.k)
        # normalize 0-1
        if self.similarity_algorithm == "dot_product":
            scores = ((scores + 1.0) / 2.0)
        # ranking
        corpus_container = self.vector_data.get_df_doc_feature()[["doc_id", "feature_name", "chunk_id"]]
        corpus_container["scores"] = 0.0
        corpus_container["scores"] = corpus_container["scores"].astype("float32")
        return self.ranker(corpus_container, scores, indicies, excluding_zero_score)
