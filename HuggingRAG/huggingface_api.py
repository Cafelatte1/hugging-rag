import pandas as pd
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

"""## Generation with Retrieval Documents"""

class HuggingFaceAPI():
    def __init__(self, model_id, tokenizer_max_length, vector_data, vector_embedding, vector_store, quantization_params="auto", device="cpu"):
        self.vector_data = vector_data
        self.vector_embedding = vector_embedding
        self.vector_store = vector_store
        self.model_id = model_id
        self.device = torch.device("cuda" if device == "gpu" else device)
        self.max_length = tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, truncation_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer_params = {
            "max_length": self.max_length,
            "padding": False,
            "truncation": True,
            "return_attention_mask": True,
            "return_token_type_ids": False,
            "return_tensors": "pt"
        }
        if quantization_params is not None:
            if quantization_params == "auto":
                quantization_params = BitsAndBytesConfig(
                    # 4bit quantization
                    load_in_4bit=True,
                    # set data type in saving the weights
                    bnb_4bit_quant_type="nf4",
                    # use double quantization
                    bnb_4bit_use_double_quant=True,
                    # set data type in calculating the weights
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_params, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.to(self.device)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model.config.use_cache = True

    def tokenize(self, x):
        tokens = self.tokenizer.encode_plus(
            x, **self.tokenizer_params,
        )
        return tokens

    def create_prompt_template(self, lang="kor"):
        if lang in ["kr", "kor"]:
            prompt = """지시문: 검색된 문서들을 참고하여 요청에 알맞는 답변을 해주세요.
검색된 문서들은 ``` 구분자 안에 [Document N] 형식으로 있습니다.
모르는 요청이면 '잘 모르겠습니다.'라고 답변해주세요.

검색된 문서
```
{context}
```

요청: {question}

답변: """
        else:
            prompt = """Instructions: Please refer to the searched documents to provide an appropriate response to the request.
The searched documents are in the format [Document N] within the ``` delimiter.
If you do not know the request, please respond with 'I don't know.'

Searched documents:
```
{context}
```

Request: {question}

Response: """
        return prompt

    def generate(
            self, prompt, search_query, question, doc_keyword="Document", generation_params="auto",
            num_context_docs=1, feature_length_strategy="balanced", max_feature_length=500, feature_length_threshold=95,
        ):
        if generation_params == "auto":
            generation_params = {
                "max_new_tokens": 300,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "length_penalty": 1.0,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }
            generation_params["early_stopping"] = True if generation_params["num_beams"] > 1 else False
        generation_params["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
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
        prompt = prompt.replace("{context}", context).replace("{question}", question)
        # tokenizing
        tokens = self.tokenize(prompt)
        # generate
        start_time = time.time()
        with torch.no_grad():
            gened = self.model.generate(
                **{"input_ids": tokens["input_ids"].to(self.device), "attention_mask": tokens["attention_mask"].to(self.device)},
                **generation_params,
            )
        response = self.tokenizer.batch_decode(gened, skip_special_tokens=True)[0]
        end_time = time.time()
        # decoding
        output = {
            "retrieval_docs": retrieval_docs,
            "response": response,
            "inference_runtime": round(end_time - start_time, 3),
        }
        return output
