import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Model, GPT2Config
# initialize tokenizer and model from pretrained GPT2 model

config = GPT2Config()
config.output_attentions = True
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model(config)
max_length = model.config.n_positions

sequence = open('one_last_swirl_short_word_list.txt', "r").read()
words = sequence.split()
encodings = tokenizer.encode(' '.join(words), return_tensors='pt')

if len(words) > max_length:
    for i in tqdm(range(len(words) - max_length + 1)):
        window = torch.narrow(encodings, 1, i, max_length)
        activations = model.forward(window, output_hidden_states=True)
