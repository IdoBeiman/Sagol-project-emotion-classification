# -*- coding: utf-8 -*-
"""
Last update on Mon May  3 2021
@author: Tamara
"""
import sys
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer
import numpy as np

###############################################################################
########################### Researcher Parameters #############################
subjects_to_remove = ".\\input\\subjects_to_remove.txt"
export_dir = "./"
transcription_dir = ".\\input\\transcriptions"
annotation_dir  = ".\\input\\firebase_output"
pretrained = '.\\input\\pretrained\\bert-base-uncased'
sentiment = "Sadness"
drop_single_subject_data = True 
###############################################################################
###############################################################################

def check_researcher_params():
    usage_messege = """Usage error! This script requires the following paramenters:
        subjects_to_remove - path to a text file with names of subjects to remove
        export_dir - path of a dir to which the final files should be exported to
        transcription_dir - location of the transcription files to the stories
        annotation_dir - location of the single subject annotation files
        pretrained - location to a pretained 'trasformers-huggininface' model for embedding 
        sentiment - the sentiment you wish to extract the annotation for
        drop_single_subject_data - (True/False). Do you wish the final result won't or will include the responses of the indevidual subjects
    """
    try:
        subjects_to_remove
        export_dir
        transcription_dir
        annotation_dir
        pretrained
        sentiment
        drop_single_subject_data 
    except NameError:
        print(usage_messege)
        sys.exit()
    
    try:
        test_tokenizer = BertTokenizer.from_pretrained(pretrained)
    except ValueError:
        print("Pretrained model is not found! Plaese download a pretrained model for embedding and provide it's path")
        sys.exit()
    
    check_paths([export_dir,transcription_dir,annotation_dir])


def is_valid_dir(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        return False
    return True

def check_paths(paths):
    for path in paths:
        if (not is_valid_dir(path)):
            print(f"'{path}' is not a valid directory!")
            exit()

def get_subs_to_skip():
    try:
        text_file = open(subjects_to_remove, "r")
        return text_file.read().split(',')
    except FileNotFoundError:
        print("Couldn't find a 'subjects_to_remove' file -> no subject will be skipped")  
        return []

def single_story_aggregate(story_dir):
    story_df = pd.DataFrame()
    subject_names=[]
    for subfile in os.listdir(os.path.join(annotation_dir,story_dir)):
        if not (subfile.endswith("_continuous.csv")):
            continue
        subname = subfile.replace('_continuous.csv', '')
        if subname in subs_to_skip:
            continue
        subject_names.append(subname)
        sub_df = pd.read_csv(os.path.join(annotation_dir,story_dir,subfile),index_col=(0))
        sub_df = sub_df.reset_index()
                
        ## we only care about the answers to the relevant sentiment for each subject
        sub_df = sub_df.rename({sentiment:subname, "index":"timepoint"}, axis='columns')
                
        story_df = pd.concat([story_df, sub_df[subname]],axis=1)
              
        story_df['audio_name'] = story_dir
        story_df["timepoint"] = sub_df["timepoint"]
    
    return story_df,subject_names

def add_aggregated_scores(story_df):
    """ 
    This is the function to edit if you want to add other aggregation methods!
    Currently, getting the mean and a binary score based on it
    """
    
    df_copy = story_df.copy(deep=True)
    df_copy.drop(['timepoint', 'audio_name'],axis=1,inplace=True)
    
    # add mean
    story_df['mean'] = df_copy.mean(axis=1)
    
    # add binary score
    binary_cutoff = 3
    conditions = [(story_df['mean'] <= binary_cutoff), (story_df['mean'] > binary_cutoff)]
    values = [1,2]

    # create a new column and use np.select to assign values to it using our lists as arguments
    story_df['binary'] = np.select(conditions, values)

    
    return story_df
    
def get_subject_aggregated_data():
    print('Aggregating all responses...')
    full_df = pd.DataFrame()
    subs = []
    for story_dir in os.listdir(annotation_dir):
        story_df,subject_names = single_story_aggregate(story_dir)
        story_df = add_aggregated_scores(story_df)
        full_df = pd.concat([full_df, story_df],axis=0)
        subs.extend(subject_names)
    
    if drop_single_subject_data:
        full_df.drop(columns=subs,inplace=True)
            
    return full_df.reset_index(drop=True)
    
def get_target_times_from_sub_annot(subject_annot, audio_names):
    target_times = {}
    for story in audio_names:
        story_df = subject_annot[subject_annot["audio_name"]==story]
    
        target_times[story] = story_df.timepoint.values.tolist()
    return target_times


def get_transcribed_segements(target_times,audio_names,transcription_dir):
    print("Getting segment trascription...")
    df = pd.DataFrame(columns=['segment_offset','text','audio_name'])
    for story in audio_names:
        filepath = os.path.join(transcription_dir, f"{story}_transcription.csv")
        #transciption_by_words = pd.read_csv(filepath, names=["start","end","word"],header=)
        transciption_by_words = pd.read_csv(filepath, header=0)
        story_target_times = target_times[story]
        
        target_time_iter = iter(story_target_times)
        current_endpoint = next(target_time_iter)
        segment = ""
        for index, row in transciption_by_words.iterrows():
            if (row['end'] <= current_endpoint):
                segment += row['word']+" "
            else:
                #print(f"NEW TIMEPOINT {current_endpoint}")
                df = df.append({"segment_offset":current_endpoint, 
                                "text":segment,
                                "audio_name":story},ignore_index=True)
                #print(segment)
                #print("~~~~~")
                segment = row['word']+" "
                try:
                    current_endpoint = next(target_time_iter)
                except StopIteration:
                    print("Notice! Last part of the story not included in any segment based on target times")
        #print(segment)
        df = df.append({"segment_offset":current_endpoint, 
                                "text":segment,
                                "audio_name":story},ignore_index=True)
        
    return df

def segments_bert_embedding(segments_text):
    print("Creating segment embedding...")
    df_embeddings = pd.DataFrame()
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    for index, row in segments_text.iterrows():
        embeding_text =  row['text']       
        encoded = tokenizer.encode_plus(
        text=row['text'],  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = len(row['text'].split()),  # maximum length of a sentence
        truncation = True,
        padding='max_length',
        #pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'pt'  # ask the function to return PyTorch tensors
        )

    # Load pre-trained model (weights)
        model = BertModel.from_pretrained(pretrained,
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                          )

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.
        with torch.no_grad():
            outputs = model(encoded['input_ids'], encoded['attention_mask'])
            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding_np = torch.mean(token_vecs, dim=0).numpy()
        sentence_embedding_df = pd.DataFrame(sentence_embedding_np)
        df_embeddings = df_embeddings.append(sentence_embedding_df.T, ignore_index=True )

    return df_embeddings


###############################################################################
###############################################################################

check_researcher_params()
subs_to_skip = get_subs_to_skip()
subject_annot = get_subject_aggregated_data()
audio_names = subject_annot.audio_name.unique()
target_times  = get_target_times_from_sub_annot(subject_annot, audio_names)
transcribed_segements = get_transcribed_segements(target_times,audio_names, transcription_dir)
embedded_segments = segments_bert_embedding(transcribed_segements)
full = pd.concat([subject_annot, embedded_segments], axis = 1)
full.text= transcribed_segements['text']
full.to_csv(os.path.join(export_dir,f"{sentiment}_ML_input.csv"), index=False)

