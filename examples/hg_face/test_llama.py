# coding=utf8 

from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import LlamaForCausalLM, LlamaTokenizerFast

voc_path= "/workspace/data/llama_hg/7B"

# import ipdb; ipdb.set_trace()
tokenizer = LlamaTokenizer.from_pretrained(f"{voc_path}")
model = LlamaForCausalLM.from_pretrained("/workspace/data/llama_hg/7B")

import ipdb; ipdb.set_trace()

