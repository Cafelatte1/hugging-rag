import pandas as pd
import numpy as np
import time
from openai import OpenAI

"""## Generation with Retrieval Documents"""

class ChatGPTAPI():
    def __init__(self, model_id, max_len, vector_data, vector_embedding, vector_store):
        self.vector_data = vector_data
        self.vector_embedding = vector_embedding
        self.vector_store = vector_store
        self.model_id = model_id
        self.max_len = max_len
        self.client = OpenAI()

    def create_prompt_template(self, lang="kor"):
        if lang in ["kr", "kor"]:
            prompt = {}
            prompt["instruction"] = {
                "role": "system",
                "content": """검색된 문서들을 참고하여 요청에 알맞는 답변을 해주세요.
검색된 문서들은 ``` 구분자 안에 [Document N] 형식으로 있습니다.
모르는 요청이면 '잘 모르겠습니다.'라고 답변해주세요."""
            }
            prompt["request"] = {
                "role": "user",
                "content": """검색된 문서
```
{context}
```

요청: {question}"""
            }
        else:
            prompt = {}
            prompt["instruction"] = {
                "role": "system",
                "content": """Please refer to the searched documents to provide an appropriate response to the request.
The searched documents are in the format [Document N] within the ``` delimiter.
If you do not know the request, please respond with 'I don't know.'."""
            }
            prompt["request"] = {
                "role": "user",
                "content": """Searched documents:
```
{context}
```

Request: {question}"""
            }
        return prompt

    def generate(
            self, prompt, search_query, question, doc_keyword="Document",
            num_context_docs=1, feature_length_strategy="balanced", max_feature_length=500, feature_length_threshold=95,
        ):
        # retrieval
        retrieval_docs = self.vector_store.search(self.vector_embedding.get_vectorembedding(search_query))
        # create context from retrieved documents
        feature_names = self.vector_data.get_df_doc().columns
        context = []
        if feature_length_strategy == "balanced":
            feature_lengths = np.array([np.percentile(self.vector_data.get_df_doc()[col].apply(len), feature_length_threshold) for col in feature_names])
            feature_lengths = ((feature_lengths / (feature_lengths.sum() + 1e-7)) * max_feature_length).astype("int32")
            for idx, doc_id in enumerate(retrieval_docs["doc_id"].iloc[:num_context_docs]):
                context.append(f"[{doc_keyword} {idx+1}]\n" + "\n".join([f"{k.split('_')[-1]}: {v[:max_len]}" for max_len, (k, v) in zip(feature_lengths, self.vector_data.get_df_doc().loc[doc_id].items())]))
        else:
            feature_lengths = (np.array(1 / (len(feature_names) + 1e-7) ) * max_feature_length).astype("int32")
            for idx, doc_id in enumerate(retrieval_docs["doc_id"].iloc[:num_context_docs]):
                context.append(f"[{doc_keyword} {idx+1}]\n" + "\n".join([f"{k.split('_')[-1]}: {v[:max_len]}" for max_len, (k, v) in zip([max_feature_length] * len(feature_names), self.vector_data.get_df_doc().loc[doc_id].items())]))
        # cut text with max value
        context = "\n".join(context)
        # create prompt
        prompt = prompt["request"]["content"].replace("{context}", context).replace("{question}", question)
        prompt["request"]["content"] = prompt["request"]["content"][:self.max_len]
        # generate
        start_time = time.time()
        gened = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                prompt["instruction"],
                prompt["request"],
            ]
        )
        response = gened.choices[0].message
        end_time = time.time()
        # decoding
        output = {
            "retrieval_docs": retrieval_docs,
            "response": response,
            "inference_runtime": round(end_time - start_time, 3),
        }
        return output
