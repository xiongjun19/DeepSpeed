# coding=utf8

import os
import torch
import deepspeed
import argparse
from transformers import pipeline
from transformers import AutoTokenizer


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


def main(args):
    model = load_model(args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b",
                                              padding_side="left")
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

    ds_engine = deepspeed.initialize(model,
                                     config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    input_sentences = [
        "DeepSpeed is a machine learning framework",
        "He is working on",
        "He has a",
        "He got all",
        "Everyone is happy and I can",
        "The new movie that got Oscar this year",
        "In the far far distance from our galaxy,",
        "Peace is the only way",
    ]
    bs = 2
    inputs = input_sentences[:bs]
    generate_kwargs = dict(max_new_tokens=10, do_sample=False)
    output = generate(inputs, tokenizer, model, generate_kwargs)
    print("output is: ")
    print(output)


def generate(inputs, tokenizer, model, generate_kwargs):
    """returns a list of zipped inputs, outputs and number of new tokens"""
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


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

