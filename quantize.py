from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
QUANTIZE_METHOD = 'gptq'

pretrained_model_dir = model_name
quantized_model_dir = 'quantized_model/'+model_name.split('/')[-1] + f'_{QUANTIZE_METHOD}_quantized'


calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_name, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=8)

model.save(quantized_model_dir)
















# import logging
# import json
# import random
# import time
# from argparse import ArgumentParser
# import torch
# from datasets import Dataset
# from transformers import AutoTokenizer, TextGenerationPipeline
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from awq import AutoAWQForCausalLM
# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
# )

# QUANTIZE_METHOD = ['gptq','awq'][1]  # 0 for gptq, 1 for awq

# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'

# pretrained_model_dir = model_name
# quantized_model_dir = 'quantized_model/'+model_name.split('/')[-1] + f'_{QUANTIZE_METHOD}_quantized'

# import numpy as np
# import torch
# import torch.nn as nn

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# # os.makedirs(quantized_model_dir, exist_ok=True)
# def get_wikitext2(nsamples, seed, seqlen, model):
#     from datasets import load_dataset

#     traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
#     testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

#     from transformers import AutoTokenizer

#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
#     except Exception:
#         tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
#     trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
#     testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

#     import random

#     random.seed(seed)
#     np.random.seed(0)
#     torch.random.manual_seed(0)

#     traindataset = []
#     for _ in range(nsamples):
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         attention_mask = torch.ones_like(inp)
#         traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
#     return traindataset, testenc


# def main():
#     if QUANTIZE_METHOD == 'gptq':
#         # GPTQ quantization
#         # 使用较短的序列长度，避免 RoPE 维度问题
#         traindataset, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)

#         quantize_config = BaseQuantizeConfig(
#             bits=8,  # quantize model to 8-bit
#             group_size=128,  # it is recommended to set the value to 128
#             desc_act=False,  # desc_act and group size only works on triton
#         )

#         # load un-quantized model, the model will always be force loaded into cpu
#         # 添加 trust_remote_code 和 use_fast tokenizer 配置
#         model = AutoGPTQForCausalLM.from_pretrained(
#             pretrained_model_dir, 
#             quantize_config,
#             trust_remote_code=True,
#             use_safetensors=True
#         )

#         # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
#         # with value under torch.LongTensor type.
#         model.quantize(traindataset, use_triton=False)

#         # save quantized model
#         model.save_quantized(quantized_model_dir)

#         # save quantized model using safetensors
#         model.save_quantized(quantized_model_dir, use_safetensors=True)

#         print(f'Model is quantized with GPTQ and saved at "{quantized_model_dir}"')
        
#         # load quantized model, currently only support cpu or single gpu
#         # model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)
    
#     elif QUANTIZE_METHOD == 'awq':
#         # AWQ quantization
#         quant_config = {
#             "zero_point": True,
#             "q_group_size": 128,
#             "w_bit": 4,
#             "version": "GEMM"
#         }

#         # Load model
#         model = AutoAWQForCausalLM.from_pretrained(
#             pretrained_model_dir,
#             trust_remote_code=True
#         )
#         tokenizer = AutoTokenizer.from_pretrained(
#             pretrained_model_dir,
#             trust_remote_code=True
#         )

#         # Quantize
#         model.quantize(tokenizer, quant_config=quant_config)

#         # Save quantized model
#         model.save_quantized(quantized_model_dir)
#         tokenizer.save_pretrained(quantized_model_dir)

#         print(f'Model is quantized with AWQ and saved at "{quantized_model_dir}"')
    
#     else:
#         raise ValueError(f"Unknown quantization method: {QUANTIZE_METHOD}. Choose 'gptq' or 'awq'.")


# if __name__ == "__main__":
#     import logging

#     logging.basicConfig(
#         format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
#         level=logging.INFO,
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     main()