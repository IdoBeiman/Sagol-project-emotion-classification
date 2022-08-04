import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

sequence = "He began his premiership by forming a five-man war cabinet which included Chamerlain as Lord President " \
           "of the Council, Labour leader Clement Attlee as Lord Privy Seal (later as Deputy Prime Minister), " \
           "Halifax as Foreign Secretary and Labour's Arthur Greenwood as a minister without portfolio. In practice,"

inputs = tokenizer.encode(sequence, return_tensors='pt')
output_1 = model.generate(inputs, max_length=200, do_sample=True)
# output_2 = model.generate(inputs, max_length=200, do_sample=True)

text_1 = tokenizer.decode(output_1[0], skip_special_tokens=True)
# text_2 = tokenizer.decode(output_2[0], skip_special_tokens=True)

print(text_1)
# print()
# print(text_2)