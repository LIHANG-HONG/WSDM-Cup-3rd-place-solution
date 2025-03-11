import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModelForCausalLM, PeftModelForSequenceClassification, get_peft_model
from lmsys.training.qwen_bi_model import Qwen2BiForSequenceClassification
from accelerate import Accelerator

class ModelManager(object):
    @staticmethod
    def load_train_model(exp_config):
        #use bf16 and FlashAttention if supported
        compute_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
        llm_config = exp_config.llm_config

        device_map = {"": Accelerator().process_index}
        if llm_config.bitsandbytes:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                llm_config.backbone,
                quantization_config=bnb_config,
                device_map=device_map,
                attn_implementation=attn_implementation,
                torch_dtype=compute_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                llm_config.backbone,
                device_map=device_map,
                attn_implementation=attn_implementation,
                torch_dtype=compute_dtype,
            )
        peft_config = None
        if llm_config.use_lora:
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':False})
            model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
            lora_params = llm_config.lora_params
            peft_config = LoraConfig(
                lora_alpha=lora_params['lora_alpha'],
                lora_dropout=lora_params['lora_dropout'],
                r=lora_params['r'],
                bias=lora_params['bias'],
                task_type=lora_params['task_type'],
                target_modules=lora_params['target_modules'],
            )
        tokenizer = AutoTokenizer.from_pretrained(
            llm_config.backbone,
            add_eos_token=True,
            add_bos_token=True
        )
        # パディングトークンが設定されていない場合、EOSトークンを設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        return model, tokenizer, peft_config

    @staticmethod
    def load_test_model(model_name_or_path, adapter_path=None):
        compute_dtype = torch.bfloat16
        if adapter_path is not None:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=compute_dtype,
            )
            model = PeftModelForCausalLM.from_pretrained(model, adapter_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=compute_dtype,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer

    @staticmethod
    def load_test_model_quant(model_name_or_path, adapter_path):
        compute_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        model = PeftModelForCausalLM.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer