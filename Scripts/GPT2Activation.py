import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Model, GPT2Config
# initialize tokenizer and model from pretrained GPT2 model

config = GPT2Config()
config.output_attentions = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model(config).to(device)
max_length = model.config.n_positions


sequence = open('one_last_swirl_short_word_list', "r").read()
words = sequence.split()
encodings = tokenizer.encode(' '.join(words), return_tensors='pt').to(device)

if len(words) > max_length:
    for i in tqdm(range(len(words) - max_length + 1)):
        window = torch.narrow(encodings, 1, i, max_length)
        activations = model.forward(window, output_hidden_states=True)
        filename = '/Users/idobe/Desktop/labProject/activations/activation_{}.pt'.format(i)
        open(filename, 'w')
        torch.save(activations.hidden_states, filename)



# max_length=1024, turncation=True)

#
# for i in range(inputs.data.size(dim=1) - 1024 + 1):
#     pass
#
#
#
# info = model.forward(inputs, output_hidden_states=True)
print('end')
# outputs = model.generate(inputs, max_length=200, do_sample=True)
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)