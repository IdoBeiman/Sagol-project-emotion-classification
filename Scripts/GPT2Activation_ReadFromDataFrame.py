import glob

import pandas as pd
import torch
from tqdm import tqdm
import re
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Model,GPT2LMHeadModel, GPT2Config
# initialize tokenizer and model from pretrained GPT2 model
emotions = ['Admiration', 'Adoration', 'Aesthetic appreciation', 'Amusement', 'Anger', 'Anxiety', 'Awe', 'Boredom',
            'Calmness', 'Confusion', 'Contempt', 'Contentment', 'Craving', 'Despair', 'Disappointment', 'Disgust',
            'Embarrassment', 'Empathic pain', 'Entrancement', 'Envy', 'Excitement', 'Fear', 'Gratitude', 'Guilt',
            'Hope', 'Horror', 'Interest', 'Irritation', 'Jealousy', 'Joy', 'Nostalgia', 'Pleasure', 'Pride',
            'Relief', 'Romance', 'Sadness', 'Satisfaction', 'Sexual desire', 'Surprise', 'Sympathy', 'Triumph',
            'Expectedness', 'Pleasantness', 'Unpleasantness', 'Goal Consistency', 'Caused by agent',
            'Intentional Action', 'Caused by Self', 'Involved Close Others', 'Control', 'Morality', 'Self Esteem',
            'Suddenness', 'Familiarity', 'Already Occurred', 'Certainty', 'Repetition', 'Coping', 'Mental States',
            'Others Knowledge', 'Bodily\Disease', 'Other People', 'Self Relevance', 'Freedom', 'Pressure',
            'Consequences', 'Danger', 'Self Involvement', 'Self Consistency', 'Relationship', 'Influence',
            'Agent vs.Situation', 'Attention', 'Safety', 'Approach', 'Arousal', 'Commitment', 'Dominance',
            'Effort', 'Fairness', 'Identity', 'Upswing']
def setup_model():
    pretrained = './models_config/pretrained/config.json'
    os.environ["CURL_CA_BUNDLE"]=""
    config = GPT2Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPT2Config.from_json_file(pretrained)
    model = GPT2Model(config=config).to(device)
    max_length = model.config.n_positions
    return (model,device)

# Output: data frame in size for [num_segments, 768] consists of the <operation> values of the neurons of #LAYER for each of the segments.
# the mean value is computed by taking the mean of each word activation
def get_activations_from_csv (dirPath ,operation,layer,is_test,sentiment):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model,device=setup_model()
    os.chdir(dirPath)
    allFiles = glob.glob('*.{}'.format("csv"))
    for path in allFiles:
        os.chdir("/data/emotion_project/idomayayuli/codeFolder/source_code/scripts/{}".format("test" if is_test else "train"))
        sequence = pd.read_csv(path)
        df_embeddings = pd.DataFrame()
        # sequence = sequence[sequence[sentiment].notna()]
        sequence.reset_index(drop=True, inplace=True)
        for index, row in sequence.iterrows():
            segment = re.sub(r' {[^}]*}','',row['text'])
            input_ids = torch.tensor(tokenizer.encode(segment, return_tensors='pt').unsqueeze(0).to(device))
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids)
                hidden_states = outputs[2]
            token_vecs = hidden_states[layer][0]
            if operation == "mean":
                sentence_embedding_np = torch.mean(token_vecs, dim=0).cpu().numpy()
            elif operation == "last_word":
                sentence_embedding_np = hidden_states[layer][0][-1].cpu().numpy()
            sentence_embedding_df = pd.DataFrame(sentence_embedding_np)
            df_embeddings = df_embeddings.append(sentence_embedding_df.T, ignore_index=True)
        for emotion in emotions:
            try:
                df_embeddings[emotion]=sequence[emotion.lower()].to_numpy()
            except:
                print("missing {} sentiment in this csv".format(emotion))
        os.chdir("/data/emotion_project/idomayayuli/codeFolder/source_code/scripts")
        df_embeddings.to_csv('tokenized_files/{}_activations_layer_{}_sub{}_operation_{}_origin_{}.csv'.format("test" if is_test else "train",layer ,"005",operation,path.split(".")[0]))
        print("finished creating activations for {}".format(path))
    return df_embeddings

setup_model()
path_train = "train"
path_test = "test"
operation= "last_word"
LAYER = 8
activations = get_activations_from_csv(path_train , operation , LAYER,False,'nostalgia')
activations = get_activations_from_csv(path_test , operation , LAYER, True,'nostalgia')
print("done")






# words = sequence.split()
#
# if len(words) > max_length:
#     for i in tqdm(range(len(words) - max_length + 1)):
#         window = torch.narrow(encodings, 1, i, max_length)
#         activations = model.forward(window, output_hidden_states=True)
#         filename = '/data/emotion_project/idomayayuli/hidden_states/activation_{}.pt'.format(i)
#         open(filename, 'w')
#         torch.save(activations.hidden_states, filename)



# max_length=1024, turncation=True)

#
# for i in range(inputs.data.size(dim=1) - 1024 + 1):
#     pass
#
#
#
# info = model.forward(inputs, output_hidden_states=True)
# print('end')
# outputs = model.generate(inputs, max_length=200, do_sample=True)
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)