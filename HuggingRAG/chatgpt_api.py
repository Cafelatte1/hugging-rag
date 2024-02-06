import pandas as pd
import numpy as np
import time
from openai import OpenAI

"""## Generation with Retrieval Documents"""

class ChatGPTAPI():
    def __init__(self, model_id, max_len, vector_embedding, vector_store):
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
                "content": """\
검색된 문서들을 참고하여 요청에 알맞는 응답을 해주세요.
검색된 문서들은 ``` 구분자 안에 [Document N] 형식으로 있습니다.
[Document N]의 세부 속성 또한 'feature: text' 형식으로 나열되어 있습니다.
모르는 요청이면 '잘 모르겠습니다.'라고 응답해주세요.
"""
            }
            prompt["request"] = {
                "role": "user",
                "content": """\
검색된 문서
```
{context}
```

요청: {question}
"""
            }
        else:
            prompt = {}
            prompt["instruction"] = {
                "role": "system",
                "content": """\
Please refer to the searched documents to provide an appropriate response to the request.
The searched documents are in the format [Document N] within the ``` delimiter.
Detailed properties of [Document N] are also listed in 'feature: text' format.
If you do not know the request, please respond with 'I don't know.'.
"""
            }
            prompt["request"] = {
                "role": "user",
                "content": """\
Searched documents
```
{context}
```

Request: {question}
"""
            }
        return prompt

    def generate(
            self, prompt, search_query, question, generation_params=None, doc_keyword="Document",
            num_context_docs=1, feature_length_strategy="balanced", max_feature_length=768, feature_length_threshold=95, reformat_output=True,
        ):
        if generation_params == "auto":
            generation_params = {
                "max_tokens": 300,
                "temperature": 0.8,
                "seed": 42,
            }
            generation_params["early_stopping"] = True if generation_params["num_beams"] > 1 else False
        # retrieval
        retrieval_docs = self.vector_store.search(self.vector_embedding.get_vector_embedding(self.vector_store.vector_data.text_preprocessor(search_query)))
        # create context from retrieved documents
        if (len(self.vector_store.vector_data.content_features) == 0):
            df_content = self.vector_store.vector_data.get_df_doc()
        else:
            df_content = self.vector_store.vector_data.get_df_doc()[self.vector_store.vector_data.content_features]
        context = []
        if feature_length_strategy == "balanced":
            feature_lengths = np.array([np.percentile(df_content[col].apply(len), feature_length_threshold) for col in df_content.columns])
            feature_lengths = ((feature_lengths / (feature_lengths.sum() + 1e-7)) * max_feature_length).astype("int32")
        else:
            feature_lengths = (np.array(1 / (len(df_content.columns) + 1e-7)) * max_feature_length).astype("int32")
        for idx, doc_id in enumerate(retrieval_docs["score_by_docs"]["doc_id"].iloc[:num_context_docs]):
            context.append(f"[{doc_keyword} {idx+1}]\n" + "\n".join([f"{k.split('_')[-1]}: {v[:max_len]}" for max_len, (k, v) in zip(feature_lengths, df_content.loc[doc_id].items())]))
        # cut text with max value
        context = "\n".join(context)
        # create prompt
        prompt["request"]["content"] = prompt["request"]["content"].replace("{context}", context).replace("{question}", question)
        prompt["request"]["content"] = prompt["request"]["content"][:self.max_len]
        # generate
        start_time = time.time()
        gened = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                prompt["instruction"],
                prompt["request"],
            ],
            **generation_params
        )
        response = gened.choices[0].message.content
        if reformat_output:
            response = "\n\n".join([
                prompt["instruction"]["content"],
                prompt["request"]["content"],
                f"Response: {response}"
            ])  
        end_time = time.time()
        # return output
        output = {
            "retrieval_docs": retrieval_docs,
            "response": response,
            "inference_runtime": round(end_time - start_time, 3),
        }
        return output
