from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch



t5_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
# model = T5FineTuner(args)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
raw_inputs= [" My daughters stupid fish is dying","He is coping with the news by making the necessary arrangements for a Burial",
"at Sea If I have flush toilet to her death is an unformed concept She is thirty four years old",
"I know am unnerved by the fish is approaching the minds eye gaze into the clear round Bowl hoping for another performance of its darting dance of Life",
"instead I see only a creature is still is the ceramic mermaid It leans against save for tiny gills that seem to be gasping This is crazy",
"I know the fish is well more than three years old cost about3 at the local pet store It is the size and color of a Dorito in the unnatural natural order of things",
"Its kind are like disposable toys We humans usually consider fish like this to be eminently flushable Im just not up to it For the record",
"I am a fish person only in the sense that I like to eat them Then why have I become emotionally attached to a pocket sized creature that lives in the Cocoon of water",
"It does not sleep in my lap We do not play fetch never once have I taken it for a walk or even the swim a satisfactory answer if AIDS me but in its BB sized eyes I see or I think I see the Panic before acceptance",
"Ive seen that before and otherwise Im never mind just know that I have become a caregiver to a3 fish that could fit in my mouth every day Now I shake out seven or eight pallets and carefully fling them into the bowl won by one aiming the brown specks",
"So they descend where the fish can eat them with minimal movement Sometimes they float just beyond his mouth and he bites and mrs Theres little I can do more than water separates us Come on eat something please I say until I remember that",
"I am pleading with a fish then come flashbacks of having said these very words before whats similar emphasis only They were associated with cans of Ensure chocolate strawberry and vanilla"]
raw_targets= ["Happy","Happy","Sad","Sad","Happy","Happy","Happy","Sad","Sad","Happy","Sad","Sad"]
# inputs = tokenizer(raw_inputs,  padding="longest", truncation=True, return_tensors="pt")
# t5_emotion_inputs = t5_tokenizer.encode('happiness </s>')
tokenized_inputs = t5_tokenizer.batch_encode_plus(raw_inputs, padding='longest', pad_to_max_length=True, return_tensors="pt")
tokenized_targets = t5_tokenizer.batch_encode_plus(raw_targets, padding='longest', pad_to_max_length=True, return_tensors="pt" )
print(tokenized_inputs)
print(tokenizer.decode(tokenized_inputs.data['input_ids'][1]))
print(tokenizer.decode(tokenized_targets.data['input_ids'][1]))
# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss

def get_dataset(tokenizer, type_path, args):
  return EmotionDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)
  