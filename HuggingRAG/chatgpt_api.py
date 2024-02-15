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
지시문에 알맞는 응답을 해주세요.\
``` 구분자 안에 검색된 문서들과 그에 대한 정보들이 있습니다.\
문서는 'Document N' 형식으로 되어 있고 세부 속성은 'Property: Content' 형식으로 되어 있습니다.\
"""
            }
            prompt["request"] = {
                "role": "user",
                "content": """\
```
{context}
```

지시문: {instruction}"""
            }
        else:
            prompt = {}
            prompt["instruction"] = {
                "role": "system",
                "content": """\
Please give an appropriate response to the instructions.\
The searched documents and information about them are contained within the ``` separator.\
    The document is in 'Document N' format and the detailed properties are in 'Property: Content' format.\
"""
            }
            prompt["request"] = {
                "role": "user",
                "content": """\
```
{context}
```

Instruction: {instruction}"""
            }
        return prompt

    def generate(
            self, prompt, search_query, instruction, generation_params=None, doc_keyword="Document",
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
        prompt["request"]["content"] = prompt["request"]["content"].replace("{context}", context).replace("{instruction}", instruction)
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
