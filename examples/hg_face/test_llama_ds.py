# coding=utf8 


import os
import argparse
import torch
import deepspeed
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TextGenerationPipeline
from transformers import Pipeline


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))


print(f"world_size: {world_size}")
print(f"local rank: {local_rank}")

class Infer(object):
   def __init__(self, model_path):
       self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
       self.model = LlamaForCausalLM.from_pretrained(model_path)
       # self.pipeline = TextGenerationPipeline(self.model, self.tokenizer, framework='pt', device=0)
      #  self.pipeline = TextGenerationPipeline(self.model, self.tokenizer, framework='pt', device=0)
      
       self.model = deepspeed.init_inference(self.model,
                                           mp_size=world_size,
                                           dtype=torch.float16,
                                           replace_with_kernel_inject=True) 
       self.pipeline = TextGenerationPipeline(self.model, self.tokenizer)

       
    
   def infer(self, sentence):
       outputs = self.pipeline(sentence, do_sample=False, min_length=10, max_length=20)
       return outputs
       

if __name__ == '__main__':
    model_path='/workspace/data/llama_hg/7B'
    infer_obj = Infer(model_path)
    input_sentence = 'hello world'
    output = infer_obj.infer(input_sentence)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("output is: ")
        print(output)



       


