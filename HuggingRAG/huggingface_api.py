import os
import pandas as pd
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import TensorDataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""## Generation with Retrieval Documents"""

class HuggingFaceAPI():
    def __init__(self, model_id, vector_embedding, vector_store, generate_params, quantization_params=None, max_length=512, device="cpu"):
        self.vector_embedding = vector_embedding
        self.vector_store = vector_store
        self.model_id = model_id
        if generate_params is True:
            generate_params = {
                "max_length": 1000,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "eos_token_id": 2,
            }
            generate_params["early_stopping"] = True if generate_params["num_beams"] > 1 else False
        self.generate_params = generate_params
        if quantization_params is True:
            quantization_params = BitsAndBytesConfig(
                # 모델을 4bit로 로딩하도록 설정합니다
                load_in_4bit=True,
                # double quantization 모드를 활성화합니다 (weight 저장과 계산을 다른 타입으로 할 수 있게 합니다)
                bnb_4bit_use_double_quant=True,
                # double quantization 모드에서 저장될 데이터 타입을 지정합니다
                bnb_4bit_quant_type="nf4",
                # double quantization 모드에서 계산에 데이터 타입을 지정합니다
                bnb_4bit_compute_dtype=torch.bfloat16,
                # set device
                device_map="auto",
            )
        self.quantization_params = quantization_params
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer_params = {
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "return_token_type_ids": False,
            "return_tensors": "pt"
        }
        if quantization_params is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_params)
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
            prompt = """지시문: 문맥을 참고하여 질문에 알맞는 답변을 해주세요. 문맥은 ```구분자 안에 있습니다. 모르는 질문이면 '잘 모르겠습니다.'라고 답변해주세요.

문맥
```
{context}
```

질문: {question}

답변: """
        else:
            prompt = """Instruction: Please reply on question referring to context. Context is in ```separator. If you don't know about question, please reply 'I don't know.'.

Context
```
{context}
```

Question: {question}

Answer: """
        return prompt

    def generate(
            self, prompt, question, vector_data, doc_keyword="document", num_context_docs=1, feature_length_strategy="balanced",
            max_context_length=1000, max_feature_length=100, feature_length_threshold=80,
        ):
        # retrieval
        retrieval_docs = self.vector_store.search(self.vector_embedding.get_vectorembedding(question))
        # create context from retrieved documents
        feature_names = vector_data.get_df_doc().columns
        context = []
        if feature_length_strategy == "balanced":
            feature_lengths = np.array([np.percentile(vector_data.get_df_doc()[col].apply(len), feature_length_threshold) for col in feature_names])
            feature_lengths = ((feature_lengths / feature_lengths.sum()) * feature_length_threshold).astype("int32")
            for idx, doc_id in enumerate(retrieval_docs["doc_id"].iloc[:num_context_docs]):
                context.append(f"[{doc_keyword}{idx+1}]\n" + "\n".join([f"{k.split('_')[-1]}: {v[:max_len]}" for max_len, (k, v) in zip(feature_lengths, vector_data.get_df_doc().loc[doc_id].items())]))
        else:
            for idx, doc_id in enumerate(retrieval_docs["doc_id"].iloc[:num_context_docs]):
                context.append(f"[{doc_keyword}{idx+1}]\n" + "\n".join([f"{k.split('_')[-1]}: {v[:max_len]}" for max_len, (k, v) in zip([max_feature_length] * len(feature_names), vector_data.get_df_doc().loc[doc_id].items())]))
        context = "\n".join(context)
        # create prompt
        prompt = prompt.replace("{context}", context).replace("{question}", question)
        # tokenizing
        tokens = self.tokenize(prompt)
        # generate
        start_time = time.time()
        with torch.no_grad():
            for batch in DataLoader(TensorDataset(tokens["input_ids"], tokens["attention_mask"]), batch_size=1, shuffle=False):
                batch[0] = batch[0].to(self.device)
                batch[1] = batch[1].to(self.device)
                gened = self.model.generate(
                    **{"input_ids": batch[0], "attention_mask": batch[1]},
                    **self.generate_params,
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

# # === EXAMPLE ===
# generate_params = {
#     "max_new_tokens": 300,
#     "num_beams": 3,
#     "do_sample": True,
#     "temperature": 0.7,
#     "top_k": 50,
#     "top_p": 0.9,
#     "length_penalty": 0.8,
#     "repetition_penalty": 1.2,
#     "no_repeat_ngram_size": 3,
#     "eos_token_id": 2,
# }
# generate_params["early_stopping"] = True if generate_params["num_beams"] > 1 else False

# # config on model for quantization
# quantization_params = BitsAndBytesConfig(
#     # 모델을 4bit로 로딩하도록 설정합니다
#     load_in_4bit=True,
#     # double quantization 모드를 활성화합니다 (weight 저장과 계산을 다른 타입으로 할 수 있게 합니다)
#     bnb_4bit_use_double_quant=True,
#     # double quantization 모드에서 저장될 데이터 타입을 지정합니다
#     bnb_4bit_quant_type="nf4",
#     # double quantization 모드에서 계산에 데이터 타입을 지정합니다
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     # set device
#     device_map="auto",
# )

# model_id = "EleutherAI/polyglot-ko-1.3b"
# llm_seq_len = 2048
# llm = HuggingFaceAPI(model_id, vector_embedding, vector_store, generate_params=generate_params, quantization_params=quantization_params, max_length=llm_seq_len)

# prompt = llm.create_prompt_template()
# question = "금은품 절도 사건 관련한 문서의 재판부 판결을 요약해주세요."
# output = llm.generate(
#     prompt, question, doc_keyword="문서",
#     feature_length_strategy="balanced", feature_length_threshold=80, max_context_length=int(llm_seq_len * 0.5),
# )

# print(output["response"])