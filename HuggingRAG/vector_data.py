import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""## Create Vector Data"""

class VectorDataContainer():
    def __init__(self, text_splitter=None):
        self.df_doc = None
        self.df_doc_feature = None
        if text_splitter is None:
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        else:
            self.text_splitter = text_splitter

    def text_preprocess(self, text):
        return " ".join(text.split())

    def get_vectordata(self, doc_id, doc_features):
        self.df_doc = pd.DataFrame(doc_features)
        self.df_doc.index = doc_id
        df_doc_feature = []
        for idx, row in tqdm(self.df_doc.iterrows(), total=len(self.df_doc)):
            for feature_name, feature_text in row.items():
                feature_text = self.text_preprocess(feature_text)
                for chunk_id, chunk in enumerate(self.text_splitter.split_text(feature_text)):
                    df_doc_feature.append({
                        "doc_id": idx,
                        "feature_name": feature_name,
                        "chunk_id": chunk_id,
                        "chunk": f"{feature_name}: {chunk}",
                    })
        self.df_doc_feature = pd.DataFrame(df_doc_feature)

    def get_df_doc(self):
        return self.df_doc

    def get_df_doc_feature(self):
        return self.df_doc_feature

    def get_chunks(self):
        return self.df_doc_feature["chunk"].to_list()

# # === EXAMPLE ===
# vector_data = VectorDataContainer(text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20))
# vector_data.get_vectordata(
#     doc_id=range(len(df)),
#     doc_features={
#         "기초사실": df["feature_기초사실"],
#     }
# )
