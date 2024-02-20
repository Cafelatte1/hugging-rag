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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", truncation_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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

    # alpaca style prompt format
    def create_prompt_template(self, lang="kor"):
        if lang in ["kr", "kor"]:
            prompt = """{bos_token}\
아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. \
입력에는 검색된 문서들과 그에 대한 정보들이 있습니다. \
문서는 'Document N' 형식으로 되어 있고 세부 속성은 'Property: Content' 형식으로 되어 있습니다. \
요청을 적절히 완료하는 응답을 작성하세요.

### 지시문:
{instruction}

### 입력:
{context}

### 응답:
{eos_token}"""
        else:
            prompt = """{bos_token}\
Below is an instruction that describes a task, paired with an input that provides further context. \
Input contains the searched documents and information about them. \
The document is in 'Document N' format and the detailed properties are in 'Property: Content' format. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{eos_token}"""
        return prompt

    def generate(
            self, prompt, search_query_list, instruction_list, generation_params="auto", batch_size=1,
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
        search_query_list = [search_query_list] if isinstance(search_query_list, str) else search_query_list
        instruction_list = [instruction_list] if isinstance(instruction_list, str) else instruction_list
        # create prompt
        prompt_list = []
        retrieval_docs_list = []
        for search_query, instruction in zip(search_query_list, instruction_list):
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
                context.append(f"Document {idx+1}\n" + "\n".join([f"{k.split('_')[-1]}: {v[:max_len]}" for max_len, (k, v) in zip(feature_lengths, df_content.loc[doc_id].items())]))
            context = "\n".join(context)
            prompt_mapper = {
                "{bos_token}": "" if self.tokenizer.bos_token is None else self.tokenizer.bos_token,
                "{instruction}": instruction,
                "{context}": context,
                "{eos_token}": "",
            }
            for k, v in prompt_mapper.items():
                prompt = prompt.replace(k, v)
            prompt_list.append(prompt)
            retrieval_docs_list.append(retrieval_docs)

        start_time = time.time()
        response_list = []
        # batch-generation
        if batch_size > 1:
            tokenizer_params = {
                "max_length": self.max_length,
                "padding": "max_length",
                "truncation": True,
                "return_token_type_ids": False,
                "add_special_tokens": False,
                "return_tensors": "pt"
            }
            # tokenizing
            tokens = self.tokenizer.batch_encode_plus(prompt_list, **tokenizer_params)
            dl = DataLoader(TensorDataset(tokens["input_ids"], tokens["attention_mask"]), batch_size=batch_size, shuffle=False)
            # generate
            with torch.no_grad():
                for batch in tqdm(dl):
                    gened = self.model.generate(
                        **{"input_ids": batch[0].to(self.device), "attention_mask": batch[1].to(self.device)},
                        **generation_params,
                    )
                    # decoding
                    response_list.extend(self.tokenizer.batch_decode(gened, skip_special_tokens=True))
                    del batch, gened
                    torch.cuda.empty_cache()
                    gc.collect()
            
        else:
            tokenizer_params = {
                "max_length": self.max_length,
                "padding": False,
                "truncation": True,
                "return_token_type_ids": False,
                "add_special_tokens": False,
                "return_tensors": "pt"
            }
            for prompt in prompt_list:
                # tokenizing
                tokens = self.tokenizer.encode_plus(prompt, **tokenizer_params)
                # generate
                with torch.no_grad():
                    gened = self.model.generate(
                        **{"input_ids": tokens["input_ids"].to(self.device), "attention_mask": tokens["attention_mask"].to(self.device)},
                        **generation_params,
                    )
                    # decoding
                    response_list.extend(self.tokenizer.batch_decode(gened, skip_special_tokens=True))
                    del gened
                    torch.cuda.empty_cache()
                    gc.collect()
        end_time = time.time()
        # return output
        output = {
            "retrieval_docs": retrieval_docs_list,
            "response": response_list,
            "inference_runtime": round(end_time - start_time, 3),
        }
        return output
