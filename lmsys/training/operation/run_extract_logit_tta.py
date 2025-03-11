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

from lmsys.training.config import ExperimentConfig
from lmsys.training.trainer import SFTChoiceTrainer
from lmsys.training.dataset import DatasetManager
from lmsys.training.model import ModelManager

os.environ["WANDB_DISABLED"] = "true"

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, keep):
        self.allowed_ids = keep
        
    def __call__(self, input_ids, scores):
        scores[self.allowed_ids] += 100
        return scores

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

def main(exp_config, filepath, model_name_or_path, adapter_path, output_dir):
    seed_everything(exp_config.seed)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(filepath)
    df = df[df['winner_tie'] == 0].reset_index(drop=True)
    df['winner_model_tmp'] = df['winner_model_a']
    df['winner_model_a'] = df['winner_model_b']
    df['winner_model_b'] = df['winner_model_tmp']
    df.drop(columns=['winner_model_tmp'], inplace=True)
    df['response_tmp'] = df['response_a']
    df['response_a'] = df['response_b']
    df['response_b'] = df['response_tmp']
    df.drop(columns=['response_tmp'], inplace=True)
    df_valid = df

    spec = importlib.util.spec_from_file_location("PromptManager","prompt/{prompt_name}/preference_prompt.py".format(prompt_name=exp_config.dataset_config.prompt_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PromptManager = module.PromptManager

    if model_name_or_path is None:
        model_name_or_path = exp_config.llm_config.backbone
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    valid_prompts, valid_answers,_ = DatasetManager.prepare_prompt(df_valid, exp_config, PromptManager, tokenizer, is_train=False)

    df_valid['whole_prompt'] = valid_prompts
    df_valid['answer'] = valid_answers
    
    keep = []
    for x in ['A','B']:
        c = tokenizer.encode(x,add_special_tokens=False)[0]
        keep.append(c)
    outputs_a = []
    outputs_b = []
    sampling_params = {
        "temperature": 0.01,
        "top_p": 0.9,
    }
    model, tokenizer = ModelManager.load_test_model(model_name_or_path, adapter_path)
    with torch.inference_mode():
        for prompt in tqdm(df_valid['whole_prompt'].values):
            batch = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model.generate(**batch, max_new_tokens = 1, **sampling_params, output_logits=True, return_dict_in_generate=True)
            logits = output["logits"][0][0, keep].to("cpu")
            outputs_a.append(logits[0].item())
            outputs_b.append(logits[1].item())

    df_valid['logit_a'] = outputs_a
    df_valid['logit_b'] = outputs_b
    df_valid[['id','logit_a','logit_b']].to_parquet(os.path.join(output_dir, f'{exp_config.exp_name}_logit_tta.parquet'))
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--model-name-or-path', type=str, required=False, default=None)
    parser.add_argument('--adapter-path', type=str, required=False, default=None)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    main(exp_config, args.filepath, args.model_name_or_path, args.adapter_path, args.output_dir)