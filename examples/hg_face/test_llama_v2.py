# coding=utf8 


import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TextGenerationPipeline
from transformers import Pipeline

class Infer(object):
   def __init__(self, model_path):
       self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
       self.model = LlamaForCausalLM.from_pretrained(model_path)
       # self.pipeline = TextGenerationPipeline(self.model, self.tokenizer, framework='pt', device=0)
       self.pipeline = TextGenerationPipeline(self.model, self.tokenizer, framework='pt', device=-1)
       
    
   def infer(self, sentence):
       outputs = self.pipeline(sentence, do_sample=False)
       return outputs
       

if __name__ == '__main__':
    model_path='/workspace/data/llama_hg/7B'
    infer_obj = Infer(model_path)
    input_sentence = 'hello world'
    output = infer_obj.infer(input_sentence)
    print(output)



       


