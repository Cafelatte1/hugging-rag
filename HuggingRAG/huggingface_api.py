import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

"""## Generation with Retrieval Documents"""

class HuggingFaceAPI():
    def __init__(self, model_id, tokenizer_max_length, vector_embedding, vector_store, quantization_params="auto", device="cpu"):
        self.vector_embedding = vector_embedding
        self.vector_store = vector_store
        self.model_id = model_id
        self.device = torch.device("cuda" if device == "gpu" else device)
        self.max_length = tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", runcation_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer_params = {
            "max_length": self.max_length,
            "padding": "max_length",
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
        tokens = self.tokenizer.batch_encode_plus(
            x, **self.tokenizer_params,
        )
        return tokens

    def create_prompt_template(self, lang="kor"):
        if lang in ["kr", "kor"]:
            prompt = """지시문: 검색된 문서들을 참고하여 요청에 알맞는 응답을 해주세요.
검색된 문서들은 ``` 구분자 안에 [Document N] 형식으로 있습니다.
[Document N]의 세부 속성 또한 'feature: text' 형식으로 나열되어 있습니다.
모르는 요청이면 '잘 모르겠습니다.'라고 응답해주세요.

검색된 문서
```
{context}
```

요청: {question}

응답: """
        else:
            prompt = """Instructions: Please refer to the searched documents to provide an appropriate response to the request.
The searched documents are in the format [Document N] within the ``` delimiter.
Detailed properties of [Document N] are also listed in 'feature: text' format.
If you do not know the request, please respond with 'I don't know.'

Searched documents
```
{context}
```

Request: {question}

Response: """
        return prompt

    def generate(
            self, prompt, search_query_list, question_list, doc_keyword="Document", generation_params="auto", batch_size=1,
            num_context_docs=1, feature_length_strategy="balanced", max_feature_length=768, feature_length_threshold=95,
        ):
        if generation_params == "auto":
            generation_params = {
                "max_new_tokens": 300,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "length_penalty": 1.2,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }
            generation_params["early_stopping"] = True if generation_params["num_beams"] > 1 else False
        search_query = [search_query] if isinstance(search_query, str) else search_query
        question = [question] if isinstance(question, str) else question
        # create prompt
        prompt_list = []
        retrieval_docs_list = []
        for search_query, question in zip(search_query_list, question_list):
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
            prompt_list.append(prompt.replace("{context}", context).replace("{question}", question))
            retrieval_docs_list.append(retrieval_docs)
        # tokenizing
        tokens = self.tokenize(prompt_list)
        dl = DataLoader(TensorDataset(tokens["input_ids"], tokens["attention_mask"]), batch_size=batch_size, shuffle=False)
        # generate
        start_time = time.time()
        response_list = []
        with torch.no_grad():
            for batch in tqdm(dl):
                gened = self.model.generate(
                    **{"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device)},
                    **generation_params,
                )
                del batch, gened
                torch.cuda.empty_cache()
                gc.collect()
                # decoding
                response_list.extend(self.tokenizer.batch_decode(gened, skip_special_tokens=True))
        end_time = time.time()
        # return output
        output = {
            "retrieval_docs": retrieval_docs_list,
            "response": response_list,
            "inference_runtime": round(end_time - start_time, 3),
        }
        return output
