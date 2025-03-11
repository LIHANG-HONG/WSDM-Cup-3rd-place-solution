import torch
from vllm import LLM
import argparse

def main(model_path):
    num_gpu = torch.cuda.device_count()
    llm = LLM(
        model_path,
        tensor_parallel_size=num_gpu,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2048,
        disable_log_stats=True,
        max_num_seqs=512,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args.model_path)