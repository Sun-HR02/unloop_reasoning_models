import logging
import json
import random
import time
from argparse import ArgumentParser
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

QUANTIZE_METHOD = ['gptq','awq'][0]

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

pretrained_model_dir = model_name
quantized_model_dir = 'quantized_model/'+model_name.split('/')[-1] + f'{QUANTIZE_METHOD}_quantized'

import numpy as np
import torch
import torch.nn as nn

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


def main():
    # 使用较短的序列长度，避免 RoPE 维度问题
    traindataset, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)

    quantize_config = BaseQuantizeConfig(
        bits=8,  # quantize model to 8-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )

    # load un-quantized model, the model will always be force loaded into cpu
    # 添加 trust_remote_code 和 use_fast tokenizer 配置
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_dir, 
        quantize_config,
        trust_remote_code=True,
        use_safetensors=True
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset, use_triton=False)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)

    # load quantized model, currently only support cpu or single gpu
    # model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()