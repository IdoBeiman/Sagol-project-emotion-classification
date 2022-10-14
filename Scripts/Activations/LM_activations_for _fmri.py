

import pandas as pd
import torch, re, os, glob
import numpy as np
from praatio import textgrid
from transformers import GPT2Model, BertModel, T5Model, GPT2Config,GPT2Tokenizer, BertTokenizer, BertConfig, T5Tokenizer, T5Config



def setup_model(model_name, preload_config=False, pretrained_model_path=None):
    """
    Setup model and tokenizer
    parameters:
        model_name: name of model to load
        preload_config: whether to use a local config json file
        pretrained_model_path: path to pretrained model - currently only used for GPT2
    """
    os.environ["CURL_CA_BUNDLE"] = ""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name.lower() == "gpt" or model_name.lower() == "gpt2":
        if preload_config:
            pretrained = './models_config/pretrained/config.json'
            config = GPT2Config()
            config = GPT2Config.from_json_file(pretrained)
            model = GPT2Model(config=config).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            model = GPT2Model.from_pretrained(model_name).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_name.lower() == "bert":
        if preload_config:
            pretrained = './models_config/pretrained/bert_config.json'
            config = BertConfig()
            config = BertConfig.from_json_file(pretrained)
            model = BertModel(config=config).to(device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            model = BertModel.from_pretrained('bert-base-uncased').to(device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name.lower() == "t5":
        if preload_config:
            pretrained = './models_config/pretrained/config.json'
            config = T5Config()
            config = T5Config.from_json_file(pretrained)
            model = T5Model(config=config).to(device)
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
        else:
            model = T5Model.from_pretrained('t5-base').to(device)
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
    elif model_name.lower() == "gpt2-pretrained":
        pretrained = f"{pretrained_model_path}/config.json"
        config = GPT2Config()
        config = GPT2Config.from_json_file(pretrained)
        model = GPT2Model(config=config).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_path)
    return (model, tokenizer, device)


def add_textgrid_time_to_activations(df, textgrid_path, gpt_toks=None, split_ws=None):
    """
    Add time information to activations
    parameters:
        df: dataframe of activations
        textgrid_path: path to textgrid file
    optional parameters:
        gpt_toks: list of tokens from gpt tokenizer
        split_ws: list of indices of words that were split by the tokenizer
        optional params are used to remove non-matching but resolved gpt-2 tokens
    """

    # extract gpt tokens for comparison
    if gpt_toks is not None:
        LM_toks = []
        pre_split_ws = [i - 1 for i in split_ws if i - 1 not in split_ws]
        for i,w in enumerate(gpt_toks):
            if i not in split_ws and i not in pre_split_ws:
                LM_toks.append(w.strip(' Ġ'))
            elif i not in split_ws and i in pre_split_ws:
                LM_toks.append(' ')

    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    word_tier = tg.tierDict['words']
    tg_df = pd.DataFrame([(start, end, label) for (start, end, label) in word_tier.entryList],
                         columns=['start', 'end', 'label'])
    # drop rows with non-verbal markers
    drop_inds = tg_df.loc[tg_df['label'].str.contains("{")].index.to_list()
    tg_df.drop(drop_inds, inplace=True)
    tg_df.reset_index(inplace=True)
    # assert that the number of words in the textgrid is the same as the number of words in the LM activationssize
    # assert tg_df.shape[0] == df.shape[0], "Number of words in textgrid and LM activations does not match"
    if gpt_toks is not None:
        assert tg_df.shape[0] == len(LM_toks), "Number of words in textgrid and LM tokens does not match"
        spaces = [i for i, x in enumerate(LM_toks) if x == ' ']
        tmp_tg = tg_df['label'].to_list()
        for i in sorted(spaces, reverse=True):
            del tmp_tg[i]
            del LM_toks[i]
        assert tmp_tg == LM_toks, "LM tokens and textgrid words do not match"
    df['start'] = tg_df['start']
    df['end'] = tg_df['end']
    """
    # If asserts fail - execute this part to see where the mismatch is
    spaces = [i for i, x in enumerate(LM_toks) if x == ' ']
    tmp_tg = tg_df['label'].to_list()
    for i in sorted(spaces, reverse=True):
        del tmp_tg[i]
        del LM_toks[i]
        
    from difflib import Differ
    d = list(Differ().compare(tmp_tg, LM_toks))
    # words in csv that are not in textgrid
    adds = [(i, x) for i,x in enumerate(d) if x[0] == '+']
    # words in textgrid but not in csv input file
    subts = [(i, x) for i,x in enumerate(d) if x[0] == '-']
    
    """
    return df


def sliding_window_tensor(ind, token_tensor, context_length = 25):
    """
    Get a sliding window of tokens for a given index
    parameters:
        ind: index of token to get context for
        context_length: number of tokens to get before and after the given index
        token_tensor: tensor of tokens
    """
    # exs: 0: 0 1; 1: 0 1 2;

    dims_token_tensor = len(token_tensor.shape)
    if ind - context_length < 0:
        start = 0
        end = ind+1
    else:
        start = ind - context_length
        if ind + context_length > token_tensor.shape[-1]+1:
            end = token_tensor.shape[-1]+1
        else:
            end = ind+1
    if dims_token_tensor == 3:
        return token_tensor[:, :, start:end]
        #return list(range(start, end))
    elif dims_token_tensor == 2:
        return token_tensor[:, start:end]
        #return list(range(start, end))


# Output: data frame in size for [num_segments, 768] consists of the <operation> values of the neurons of #LAYER for each of the segments.
# the mean value is computed by taking the mean of each word activation
def get_activations_from_csv(dir_path, sub, out_dir, grid_path, operation, layer, model_name):
    model, tokenizer, device = setup_model(model_name)
    # csv files are transcripts split to segments + their emotion labels
    all_files = glob.glob(f"{dir_path}/sub-{sub}/sub-{sub}*.csv")
    merged_df = pd.DataFrame()
    i=0
    for path in all_files:
        # debugging
        """
        if i<22:
            i+=1
            continue
        """
        i+=1
        print(f"processing file {i}/{len(all_files)}")
        sequence = pd.read_csv(path)
        df_embeddings = pd.DataFrame()
        segments_lengths = []
        episode_name = path.split('_')[1].split(".")[0]
        sequence.reset_index(drop=True, inplace=True)
        model.eval()
        for index, row in sequence.iterrows():
            segment = re.sub(r' {[^}]*}', '', row['text']) # remove manual markings like {lg} for laughter
            if("gpt" in model_name.lower()):
                encoded_segment = torch.tensor(tokenizer.encode(
                    segment, return_tensors='pt', add_prefix_space=True).unsqueeze(0).to(device))
            else:
                encoded_segment = torch.tensor(tokenizer.encode(segment, return_tensors='pt').unsqueeze(0).to(device))
            if index == 0:
                ep_toks = encoded_segment
            else:
                ep_toks = torch.cat((ep_toks, encoded_segment), dim=len(encoded_segment.shape)-1)
        # debugging
        #input_ids=[]
        for j in range(1,ep_toks.shape[-1]+1):
            input_ids = sliding_window_tensor(j, ep_toks)
            # debugging! delete!
            #input_ids.append(sliding_window_tensor(j, ep_toks))
            #continue
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                if j==1:
                    ep_activations = outputs.hidden_states[layer][0][-1].cpu().numpy()
                else:
                    # activation matrix of shape words*units
                    ep_activations = np.vstack((ep_activations, outputs.hidden_states[layer][0][-1].cpu().numpy()))
        sentence_embedding_np = ep_activations.copy()
        gpt_tok = tokenizer.convert_ids_to_tokens(torch.squeeze(ep_toks).tolist())
        split_ws = [i for i, v in enumerate(gpt_tok) if not v.startswith('Ġ')]
        sentence_embedding_np = np.delete(sentence_embedding_np, split_ws, axis=0)
        sentence_embedding_df = pd.DataFrame(sentence_embedding_np)
        sentence_embedding_df["episodeName"] = episode_name
        sentence_embedding_df = sentence_embedding_df.reset_index()
        sentence_embedding_df.rename(columns={'index': 'word_index'}, inplace=True)
        df_embeddings = df_embeddings.append(sentence_embedding_df, ignore_index=True)
        # match word timing from textgrid
        # use df_embeddings, gpt_toks, textgrid path
        ep_grid_path = f'{grid_path}/sub-{sub}/{os.path.splitext(os.path.basename(path))[0]}.TextGrid'
        df_embeddings = add_textgrid_time_to_activations(df_embeddings, ep_grid_path, gpt_toks=gpt_tok, split_ws=split_ws)
        merged_df = pd.concat([merged_df, df_embeddings])

    out_path = fr"{out_dir}/sub-{sub}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # BIDS format sub-XXX_LMActivation-modelname-layer-ZZZ_sentence-OPERATION.csv
    merged_df.to_csv(f"{out_path}/sub-{sub}_LMActivation-{model_name}-layer-{layer}_sentence-{operation}.csv")
    print("finished creating activations for {}".format(path))
    return merged_df



# this script takes the csv file with segments and labels, and creates the ML Input file
# the user can modify the configuration below




# script variables
OPERATION = 'all_words'
MODEL_NAME = "gpt2"
LAYER = 9
SUB = '007'

# paths laptop
SRC_DIR = r"C:\Research\emotion\analysis\derivatives\transcriptions\segmented"
pretrained_model_path = '/data/emotion_project/transcriptions/labels_with_text/pre_train_data/pre-trained-model/'
TEXTGRID_DIR = r"C:\Research\emotion\analysis\derivatives\transcriptions\textgrids"
OUTPUT_DIR = r"C:\Research\emotion\analysis\derivatives\LM"

"""
#paths t4
SRC_DIR = ["all_data/labels_with_text/sub-006", "all_data", "all_data/labels_with_text/sub-007"]
pretrained_model_path = '/data/emotion_project/transcriptions/labels_with_text/pre_train_data/pre-trained-model/'
TEXTGRID_DIR = "/data/emotion_project/transcriptions/aligned"
"""


"""
# TODO add assert that textgrid and transcription are of same length (currently fixed for gpt)
# TODO instead of run from main - call from some emotion_vwm module?
"""
if __name__ == '__main__':
    setup_model(MODEL_NAME)
    activations = get_activations_from_csv(
        SRC_DIR, SUB, OUTPUT_DIR, TEXTGRID_DIR,
        OPERATION, LAYER, MODEL_NAME)
    print("done")

