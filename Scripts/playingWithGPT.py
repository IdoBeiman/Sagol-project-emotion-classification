import pandas as pd
# import tensorflow as tf
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
# initialize tokenizer and model from pretrained GPT2 model

config=GPT2Config()
config.output_attentions=True
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
sequence=open('one_last_swirl_short_word_list.txt',"r").read()
inputs = tokenizer(sequence, return_tensors='pt')
max_length = model.config.n_positions
stride=512
nlls = []
for i in tqdm(range(0, inputs.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, inputs.input_ids.size(1))
    trg_len = end_loc - i
    input_ids = inputs.input_ids[:,begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl.item())
# print(model.loss)

# outputs = model.generate(inputs, max_length=200, do_sample=True)
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
