import pandas as pd
from tqdm import tqdm

"""## Create Vector Data"""

def split_text_with_overlap(text, chunk_size, overlap_size, min_chunk_size):
    """
    Split a text into chunks with a specified overlap size.
    
    Parameters:
        text (str): The input text to split.
        chunk_size (int): The size of each chunk.
        overlap_size (int): The size of the overlap between chunks.
        
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if len(text[start:end]) < min_chunk_size:
            break
        chunks.append(text[start:end])
        start += chunk_size - overlap_size
    return chunks

class VectorDataContainer():
    def __init__(self, query_features=[], content_features=[], text_preprocessor=None, text_splitter=None):
        self.df_doc = None
        self.df_doc_feature = None
        # features to be searched
        self.query_features = query_features
        # features to be presented from searched documents
        self.content_features = content_features
        self.text_preprocessor = (lambda text: " ".join(text.split())) if text_preprocessor is None else text_preprocessor
        self.text_splitter = split_text_with_overlap(chunk_size=200, chunk_overlap=20, min_chunk_size=100) if text_splitter is None else text_splitter

    def get_vector_data(self, doc_id, doc_features, including_feature_name=True, feature_name_separator=":"):
        self.df_doc = pd.DataFrame(doc_features)
        self.df_doc.index = doc_id
        self.df_doc = self.df_doc.apply(lambda x: x.apply(lambda text: self.text_preprocessor(text)))
        df_doc_feature = []
        for idx, row in tqdm(self.df_doc.iterrows(), total=len(self.df_doc)):
            for feature_name, feature_text in row.items():
                if (feature_name in self.query_features) or (len(self.query_features) == 0):
                    # feature_text = self.text_preprocessor(feature_text)
                    for chunk_id, chunk in enumerate(self.text_splitter(feature_text)):
                        df_doc_feature.append({
                            "doc_id": idx,
                            "feature_name": feature_name,
                            "chunk_id": chunk_id,
                            "chunk": f"{feature_name}{feature_name_separator}{chunk}" if including_feature_name else f"{chunk}",
                        })
        self.df_doc_feature = pd.DataFrame(df_doc_feature)

    def get_df_doc(self):
        return self.df_doc

    def get_df_doc_feature(self):
        return self.df_doc_feature

    def get_chunks(self):
        return self.df_doc_feature["chunk"].to_list()
