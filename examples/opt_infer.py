# coding=utf8

import torch
import deepspeed
import argparse
from transformers import pipeline
from transformers import AutoTokenizer


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


def main(args):
    model = load_model(args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b",
                                              padding_side="left")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    model_hidden_size = model.config.hidden_size
    train_bs = 8
    ds_config = {
        "fp16": {
            "enabled": True,
        },
        "bf16": {
            "enabled": False,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": train_bs,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }

    ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)

    generator.model = deepspeed.init_inference(generator.model,
                                               config=ds_config,
                                               checkpoint=None,
                                               replace_with_kernel_inject=True)
    output = generator('DeepSpeed is', do_sample=True, min_length=10)
    print("output is: ")
    print(output)


def load_model(model_path):
    model = torch.load(model_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--local_rank', type=int,  default=0)
    parser.add_argument('--conf_path', type=str,  default=None)
    args = parser.parse_args()
    main(args)

