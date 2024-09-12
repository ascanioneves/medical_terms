import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v2", 
                                                   quantization_config=BitsAndBytesConfig(load_in_4bit=True, 
                                                                                          load_in_8bit=False,
                                                                                          bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                                                                          bnb_4bit_quant_type="nf4"))

llama_model.config.use_cache = False # Não guarda na memória cache o output das camadas computadas anteriormente
llama_model.config.pretraining_tp = 1 

llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
                                                true_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

print(llama_tokenizer)
