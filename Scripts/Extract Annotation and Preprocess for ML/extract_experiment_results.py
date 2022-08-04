

# -*- coding: utf-8 -*-
"""
Assumptions: 
        - All subjects with same story ran at the same time/with regard to the same config file and
          debreifing questions (some parts of the code loads the config and debreifing structure for
          the first subject snd apply it to all subjects of this story)
          
        - The input dir contains a text file named 'subjects_to_remove.txt' which contains the codes of 
          all the subjects we wish to skip (comma separated)
"""
import os
import sys
import argparse
import csv
import json
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pathlib import Path

###############################################################################
########################### Researcher Parameters #############################
subjects_to_remove = "./subjects_to_remove.txt"
experiment_name = "exp_1"
extract_continious = True
extract_metadata = True
extract_debreif = True
export_dir = "./output"
firebase_credentials_file = os.path.join("./", "experiment-template-firebase-adminsdk-vtljg-d688cde0e9.json")
###############################################################################
###############################################################################

def check_researcher_params():
    usage_messege = """Usage error! This script requires the following paramenters:
        experiment_name - string with the name of the expriment (i.e., the relevent collection in the firestore)
        export_dir - dir to which the experiemnt resposnes sould be extracted
        firebase_credentials_file - path of firebase credential file
        extract_continious - boolean
        extract_metadata - boolean
        extract_debreif = boolean
        subjects_to_remove - path to a text file with names of subjects to remove
        """
    try:
        experiment_name
        extract_continious
        extract_metadata
        extract_debreif
        export_dir
        firebase_credentials_file
        subjects_to_remove
    except NameError:
        print(usage_messege)
        sys.exit()
    if not os.path.exists(export_dir) or not os.path.isdir(export_dir):
        print("Invalid path given for 'export_dir'")
        sys.exit()
    if (type(extract_continious)!=bool):
        print("'extract_continious' must be boolean")
        sys.exit()
    if (type(extract_metadata)!=bool):
        print("'extract_metadata' must be boolean")
        sys.exit()
    if (type(extract_debreif)!=bool):
        print("'extract_debreif' must be boolean")
        sys.exit()
        



def setup_db(): 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=firebase_credentials_file
    cred = credentials.ApplicationDefault()
   # firebase_admin.initialize_app(cred, {  'projectId': 'experiment-template'})
    return firestore.Client() 


def get_subs_to_skip():
    path = subjects_to_remove
    try:
        text_file = open(path, "r")
        return text_file.read().split(',')
    except FileNotFoundError:
        print("Couldn't find a 'subjects_to_remove' file -> no subject will be skipped")  
        return []
    

def find_what_to_extract(s_path, sub_name, what_to_extract,subs_to_skip):
    if sub_name in subs_to_skip:
        print(f"   - Not extracting anything from {sub_name} (in 'subs_to_skip')...")
        return [False, False, False]
    wte = eval(what_to_extract)
    if (wte[0] and os.path.exists(os.path.join(s_path,f"{sub_name}_continuous.csv"))):
        wte[0] = False
        print(f"   - Not extracting continuous rating from {sub_name} (already exists)")
    if (wte[1] and os.path.exists(os.path.join(s_path,f"{sub_name}_meta.csv"))):
        wte[1] = False
        print(f"   - Not extracting metadata from {sub_name} (already exists)")
    return wte


def extract_continuous_rating_for_sub(sub_id, data, target_times, labels, save_path):
    print(target_times)
    df = pd.DataFrame(index=target_times, columns=labels+['actual_time'])
    for tp in eval(data):
        print(tp)
        df.loc[tp['target_time'], :] = {i:tp[i] for i in tp if i!='target_time'}  
    df.to_csv(os.path.join(save_path, f'{sub_id}_continuous.csv'))
    

def extract_metadata_for_sub(sub_id, data, save_path):
    m_path = os.path.join(save_path, 'metadata_and_logs')
    if not os.path.isdir(m_path):
        os.mkdir(m_path)
    with open(os.path.join(m_path, f'{sub_id}_metadata.json'), 'w') as fp:
        json.dump(data, fp)

    
def extract_debrief_for_sub(sub_id, data, save_path,df1):
    if df1.empty:
        df1 = pd.DataFrame(index=data.keys())
    df2 = pd.DataFrame(index=data.keys(),columns=[sub_id])
    for k in data.keys():
        df2.xs(k)[sub_id]=str(data[k])
    return pd.concat([df1, df2],axis=1)
    

def extract_data_from_db(db,experiment_name,subs_to_skip,export_dir,what_to_extract):   
    collection_ref = db.collection(experiment_name).document('stories')
    all_stories_refs = collection_ref.collections()
    for story_ref in all_stories_refs:
        story_name = Path(story_ref.id).stem
        print(story_name)
        s_path = os.path.join(export_dir,story_name)
        if not os.path.exists(s_path):
            os.mkdir(s_path)
        subject_stream = collection_ref.collection(story_ref.id).stream()
        target_times = []
        labels = []
        debreif_df = pd.DataFrame()    
        for sub in subject_stream:
            print(f" - {sub.id}")
            extraction_instructions = find_what_to_extract(s_path, sub.id, what_to_extract,subs_to_skip)
            data = sub.to_dict()
            if (len(target_times) == 0 or len(labels) == 0):
                target_times = data['metadata']['config_log']['timestemps']
                labels = data['metadata']['labels']
            if extraction_instructions[0]:
                extract_continuous_rating_for_sub(sub.id, data['continuous_annotation'],target_times, labels, s_path)
            if extraction_instructions[1]:
                extract_metadata_for_sub(sub.id, data['metadata'],s_path)
                if extraction_instructions[2]:
                    debreif_df = extract_debrief_for_sub(sub.id, data['debrief'],s_path,debreif_df)

                debreif_df.to_csv(os.path.join(s_path, f'{story_name}_Debrief.csv'))
        
def main(experiment_name,db):

    collnameref = db.collection(experiment_name).document('stories')
    all_stories = collnameref.collections()
    for story_ref in all_stories:
        story_name = story_ref.id
        subject_stream = collnameref.collection(story_name).stream()
        for sub in subject_stream:
            print(f" - {sub.id}")
            

check_researcher_params()
what_to_extract = str([extract_continious,extract_metadata,extract_debreif])
db = setup_db()
subs_to_skip = get_subs_to_skip()
extract_data_from_db(db,experiment_name,subs_to_skip,export_dir,what_to_extract)
