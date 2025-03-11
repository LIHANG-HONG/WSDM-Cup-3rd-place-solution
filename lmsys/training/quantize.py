import argparse
import gc
import os
import importlib
import numpy as np
import pandas as pd
import random
import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, TrainingArguments, LogitsProcessor
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from vllm import LLM, SamplingParams
from auto_round import AutoRound

from lmsys.training.config import ExperimentConfig
from lmsys.training.trainer import SFTChoiceTrainer
from lmsys.training.dataset import DatasetManager
from lmsys.training.model import ModelManager

os.environ["WANDB_DISABLED"] = "true"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)
    print('> SEEDING DONE')

def main(exp_config, filepath, fold, adapter_path, output_dir):
    seed_everything(exp_config.seed)
    df = pd.read_parquet(filepath)
    df = df[df['winner_tie'] == 0].reset_index(drop=True)
    df = df[df['author'].isin(['lmsys2','lmsys1'])].reset_index(drop=True)
    df_calib = df[df['fold'] != fold].reset_index(drop=True)

    spec = importlib.util.spec_from_file_location("PromptManager","prompt/{prompt_name}/preference_prompt.py".format(prompt_name=exp_config.dataset_config.prompt_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PromptManager = module.PromptManager

    model_name_or_path = exp_config.llm_config.backbone
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    calib_prompts, calib_answers = DatasetManager.prepare_quant_prompt(df_calib, exp_config, PromptManager, tokenizer)

    df_calib['whole_prompt'] = calib_prompts
    df_calib['answer'] = calib_answers
    calib_dataset = Dataset.from_pandas(df_calib)    
    model, tokenizer = ModelManager.load_test_model_quant(model_name_or_path, adapter_path)
    model = model.merge_and_unload()

    calib_prompts = calib_dataset['whole_prompt']
    bits, group_size, sym = 8, 128, True
    autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, dataset=calib_prompts, seqlen=6144,
                        nsamples=512,
                        iters=3000,
                        low_gpu_mem_usage=True,
                        )
    autoround.quantize()
    autoround.save_quantized(output_dir, format='auto_gptq', inplace=True)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--adapter-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    main(exp_config, args.filepath, args.fold, args.adapter_path, args.output_dir)
