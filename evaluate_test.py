
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from trl import SFTTrainer
import evaluate
from datasets import load_dataset
PEFT_MODEL = "StoneZhang/guanaco-test-l3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  #
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True,
)

# loading trained model from hugging face
config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)
# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

dataset = load_dataset(guanaco_dataset, split="train")
test_set=dataset.shuffle(seed=42).select(range(1))
perplexity = evaluate.load("perplexity", module_type="metric")
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)

references = test_set["text"]

predictions = pipe(references)

predictions = list(map(lambda prediction: prediction[0]["generated_text"], predictions))

perplexity.add_batch(predictions=predictions, references=references)

value = perplexity.compute(model_id="realshyfox/sharded-Llama-3-8B")

print("Perplexity is: ", value)
