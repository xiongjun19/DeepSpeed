
import gc
import math
import os
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig


t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")
args = parser.parse_args()




### Model loading and instantiating on GPU (via ZeRO)

model_name = args.name

print(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16

model_hidden_size = config.hidden_size
train_batch_size = 1

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

model = model.eval()
model = model.to("cuda")


### Generate

print(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

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

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

print(f"Generate args {generate_kwargs}")
inputs = input_sentences[: args.batch_size]


def generate():
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



print("*** Running generate")
t_generate_start = time.time()
pairs = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in pairs:
    print(f"{'-'*60}\nin={i}\nout={o}\n")

