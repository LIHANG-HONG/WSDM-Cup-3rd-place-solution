from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class DatasetConfig(BaseModel):
    prompt_name: str
    must_cols: List[str]
    synthetic_cols: List[str]

class LLMConfig(BaseModel):
    backbone: str
    use_lora: bool
    bitsandbytes: bool
    resume_from_checkpoint: Optional[str]
    lora_params: Optional[Dict[str, Any]]
    distil_params: Optional[List[List[Any]]]=[]

class LRSchedulerConfig(BaseModel):
    name: str
    params: Dict[str, float]

class OptimizerConfig(BaseModel):
    name: str
    params: Dict[str, float]

class ExperimentConfig(BaseModel):
    exp_type: str
    exp_name: str
    seed: int
    n_folds: int
    max_epochs: int
    num_workers: int
    train_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    dataset_config: DatasetConfig
    llm_config: LLMConfig
    lr_scheduler_config: LRSchedulerConfig
    optimizer_config: OptimizerConfig