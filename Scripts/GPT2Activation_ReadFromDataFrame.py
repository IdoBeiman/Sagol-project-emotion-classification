import glob

import numpy
import pandas as pd
import torch
from praatio import textgrid
from tqdm import tqdm
import re
import os
from transformers import GPT2Model, GPT2LMHeadModel,BertModel, T5Model, GPT2Config,GPT2Tokenizer,\
    BertTokenizer, BertConfig, T5Tokenizer, T5Config, AutoConfig, AutoModel
import numpy as np

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


def setup_model(model_name, pre_trained=None):
    os.environ["CURL_CA_BUNDLE"] = ""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pre_trained:
        config = GPT2Config.from_json_file(pre_trained + '/config.json')
        model = GPT2Model(config=config).from_pretrained(pre_trained).to(device)
        model.config.output_hidden_states = True
    elif model_name.lower() == "gpt" or model_name.lower() == "gpt2":
        pretrained = './models_config/pretrained/config.json'
        config = GPT2Config()
        config = GPT2Config.from_json_file(pretrained)
        model = GPT2Model(config=config).to(device)
    elif model_name.lower() == "bert":
        pretrained = './models_config/pretrained/bert_config.json'
        config = BertConfig()
        config = BertConfig.from_json_file(pretrained)
        model = BertModel(config=config).to(device)
    elif model_name.lower() == "t5":
        pretrained = './models_config/pretrained/config.json'
        config = T5Config()
        config = T5Config.from_json_file(pretrained)
        model = T5Model(config=config).to(device)

    # max_length = model.config.n_positions
    return (model, device)


def add_textgrid_time_to_activations(df, textgrid_path):
    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    word_tier = tg.tierDict['words']
    tg_df = pd.DataFrame([(start, end, label) for (start, end, label) in word_tier.entryList],
                         columns=['start', 'end', 'label'])
    # drop rows with non-verbal markers
    drop_inds = tg_df.loc[tg_df['label'].str.contains("{")].index.to_list()
    tg_df.drop(drop_inds, inplace=True)
    tg_df.reset_index(inplace=True)
    if tg_df.shape[0] == df.shape[0]:
        df['start'] = tg_df['start']
        df['end'] = tg_df['end']
        return df
    else:
        raise Exception('textgrid and gpt tokens have different lengths!')


# Output: data frame in size for [num_segments, 768] consists of the <operation> values of the neurons of #LAYER for each of the segments.
# the mean value is computed by taking the mean of each word activation
def get_activations_from_csv(dirPath, operation, layer, model_name, model, device, pre_trained=None):
    if model_name.lower() == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name.lower()=="gpt":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_name.lower()=="t5":
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # model, device = setup_model(model_name, pre_trained)
    os.chdir(dirPath)
    allFiles = glob.glob('*.{}'.format("csv"))
    merged_df = pd.DataFrame()
    for path in allFiles:
        os.chdir(dirPath)
        # os.chdir("/data/emotion_project/idomayayuli/codeFolder/source_code/scripts/{}".format(dirPath))
        sequence = pd.read_csv(path)
        df_embeddings = pd.DataFrame()
        segments_lengths = []
        episode_name = path.split('_')[1].split(".")[0]
        subject = path.split("_")[0]
        sequence.reset_index(drop=True, inplace=True)
        model.eval()
        # debugging line
        gpt_toks = []
        for index, row in sequence.iterrows():
            segment = re.sub(r' {[^}]*}', '', row['text']) # remove manual markings like {lg} for laughter
            if(model_name.lower() == "gpt"):
                encoded_segment = torch.tensor(tokenizer.encode(segment, return_tensors='pt', add_prefix_space=True).unsqueeze(0).to(device))
            else:
                encoded_segment = torch.tensor(tokenizer.encode(segment, return_tensors='pt').unsqueeze(0).to(device))
            if operation != "all_words":
                segments_lengths.append(len(encoded_segment[0, 0]))
                if index == 0:
                    input_ids = encoded_segment
                    if model_name.lower() == "bert" or model_name.lower() == "t5":
                        input_ids = input_ids[-1, :, :]
                else:
                    if model_name.lower() == "bert"  or model_name.lower() == "t5":
                        input_ids = torch.cat((input_ids,encoded_segment[-1,:,:]),1) # concat in the second dim
                    else:
                        input_ids = torch.cat((input_ids,encoded_segment),2) # concat in the third dim
            else:
                input_ids = encoded_segment
            input_ids_length = len(input_ids[0][0]) if model_name.lower() == "gpt" else len(input_ids[0])
            while (input_ids_length > model.config.n_positions):
                # because we concat segments, in case we surpass the
                # model's limit we will drop the first segment (or more if needed)
                num_tokens_to_remove = segments_lengths.pop(0)
                if model_name.lower() == "gpt":
                    input_ids = input_ids[:,:,num_tokens_to_remove:]
                else:
                    input_ids = input_ids[:,num_tokens_to_remove:]
                input_ids_length = len(input_ids[0][0]) if model_name.lower() == "gpt" else len(input_ids[0])
            with torch.no_grad():
                # tokenization is done for all the segments until the current one to handle context, but processing will use
                # only the segment's words
                if model_name.lower() != "t5":
                    outputs = model(input_ids)
                else:
                    outputs = model(input_ids=input_ids,decoder_input_ids=input_ids)
                hidden_states = outputs[2]
            if model_name.lower() == "bert"  or model_name.lower() == "t5":
                token_vecs = hidden_states[layer][0] if operation == "all_words" else hidden_states[layer][0][len(input_ids[0]) - len(encoded_segment):]
            else:
                token_vecs = hidden_states[layer][0] if operation == "all_words" else hidden_states[layer][0][len(input_ids[0][0]) - len(encoded_segment):]
            # take only the tokens of the current segment
            if operation == "all_words":
                sentence_embedding_np = hidden_states[layer][0].cpu().numpy()
                # identify split tokens and drop them (so that downstream dims work)
                # currently for e.g it's only 'it' activation is kept (problematic with 'don't'?)
                gpt_tok = tokenizer.convert_ids_to_tokens(tokenizer.encode(segment, add_prefix_space=True))
                split_ws = [i for i, v in enumerate(gpt_tok) if not v.startswith('Ġ')]
                sentence_embedding_np = np.delete(sentence_embedding_np, split_ws, axis=0)
                sentence_embedding_df = pd.DataFrame(sentence_embedding_np)
                sentence_embedding_df["episodeName"] = episode_name
                sentence_embedding_df["segment"] = index
                sentence_embedding_df = sentence_embedding_df.reset_index()
                sentence_embedding_df.rename(columns={'index': 'word_index'}, inplace=True)
                df_embeddings = df_embeddings.append(sentence_embedding_df, ignore_index=True)
                continue
            elif operation == "mean":
                sentence_embedding_np = torch.mean(token_vecs, dim=0).cpu().numpy()
            elif operation == "last_word":
                sentence_embedding_np = hidden_states[layer][0][-1].cpu().numpy()
            sentence_embedding_df = pd.DataFrame(sentence_embedding_np)
            df_embeddings = df_embeddings.append(sentence_embedding_df.T, ignore_index=True)
            df_embeddings["episodeName"] = episode_name
        for emotion in emotions:
            try:
                df_embeddings[emotion] = sequence[emotion.lower()].to_numpy()
            except:
                print("missing {} sentiment in this csv - {}".format(emotion, path.split(".")[0]))
        if operation == "all_words":
            # match word timing from textgrid
            # use df_embeddings, gpt_toks, textgrid path
            grid_path = f'/data/emotion_project/transcriptions/aligned/sub-005/{os.path.splitext(path)[0]}.TextGrid'
            df_embeddings = add_textgrid_time_to_activations(df_embeddings, grid_path)
        merged_df = pd.concat([merged_df, df_embeddings])
    os.chdir('C:\\Users\\Yuli\\Documents\\Uni\\sub-005-pre-trained-activations')
    # os.chdir("/data/emotion_project/idomayayuli/codeFolder/source_code/scripts")
    if pre_trained:
        merged_df.to_csv(
            'tokenized_files_new/{}_activations_layer_{}_{}_operation_{}_origin_{}_{}_model_{}_pre_trained.csv'.format("data", layer, subject,
                                                                                                           operation, "full", "with_contect", model_name))
    else:
        merged_df.to_csv(
            'tokenized_files_new/{}_activations_layer_{}_{}_operation_{}_origin_{}_{}_model_{}.csv'.format("data", layer, subject,
                                                                                                  operation, "full","with_context",model_name))
    print("finished creating activations for {}".format(path))
    return df_embeddings


path_train = "train"
path_test = "test"
# operation= "last_word"
"""
for i in range (8,12):
    for j in ["mean", "last_word"]:
        operation=j
        LAYER = i
        activations = get_activations_from_csv(path_all_data, operation, LAYER, True, 'nostalgia')
print("done")
"""

pre_trained = 'C:\\Users\\Yuli\\Documents\\Uni\\Sagol-project-emotion-classification\\Scripts\\pre_trained_model_2nd'    # path to pre_trained model, else None
all_subjects_data = ["all_data/labels_with_text/sub-006","all_data","all_data/labels_with_text/sub-007"]

for path in ['C:\\Users\\Yuli\\Documents\\Uni\\sub-005']:
    if pre_trained:
        for j in ["gpt"]:
            for i in range(12):
                for t in ["mean", "last_word"]:
                    model, device = setup_model(j, pre_trained)
                    operation = t
                    LAYER = i
                    activations = get_activations_from_csv(path, operation, LAYER, j, model, device, pre_trained)

    else:
        for j in ["t5"]: # note that t5 has only 7 layers
            for i in range(5, 7):
                for t in ["mean"]:
                    model, device = setup_model(j)
                    operation = t
                    LAYER = i
                    activations = get_activations_from_csv(path, operation, LAYER,j, model, device)
        for j in ["gpt","bert"]:
            for i in range(8, 12):
                for t in ["mean"]:
                    model, device = setup_model(j)
                    operation = t
                    LAYER = i
                    activations = get_activations_from_csv(path, operation, LAYER,j, model, device)
print("done")