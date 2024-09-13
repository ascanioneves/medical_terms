import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v2", 
                                                   quantization_config=BitsAndBytesConfig(load_in_4bit=True, 
                                                                                          load_in_8bit=False,
                                                                                          bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                                                                          bnb_4bit_quant_type="nf4"))

llama_model.config.use_cache = False # Does not save the output of the previous layers
llama_model.config.pretraining_tp = 1 

llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
                                                true_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

training_arguments = TrainingArguments(output_dir="./results", per_device_train_batch_size=4, max_steps=100)

## FINE-TUNING
llama_sft_trainer = SFTTrainer(model=llama_model, 
                               args=training_arguments,
                               train_dataset=load_dataset(path="aboonaji/wiki_medical_terms_llam2_format", split="train"),
                               tokenizer=llama_tokenizer,
                               peft_config=LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1),
                               dataset_text_field="text")

llama_sft_trainer.train()


