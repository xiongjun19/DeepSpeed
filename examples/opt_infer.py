# coding=utf8

import torch
import deepspeed
import argparse
from transformers import pipeline
from transformers import AutoTokenizer



def main(args):
    model = load_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b",
                                              padding_side="left")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    generator.model = deepspeed.init_inference(generator.model,
                                               config=args.conf_path,
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

