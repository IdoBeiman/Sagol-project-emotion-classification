import os
import shutil

import pandas as pd

subject_dir = '/data/emotion_project/transcriptions/labels_with_text/sub-005'    # path to the subject's folder
test_episodes = ['sub-005_S01E01.csv', 'sub-005_S02E02.csv', 'sub-005_S03E01P01.csv', 'sub-005_S03E05P01', 'sub-005_S04E02P01.csv', 'sub-005_S05E01P01.csv']  # should contain the full file names of the episodes that we want to use as test data.
# for example - ['sub005e01.csv', 'sub005e02.csv']
test_dir = '/data/emotion_project/idomayayuli/codeFolder/source_code/scripts/test'   # full path
train_dir = '/data/emotion_project/idomayayuli/codeFolder/source_code/scripts/train'  # full path

def divide_episodes(subject_dir, test_episodes, test_dir, train_dir):
    """
    :param subject_dir: a path to the subject dir that conatins all formated episodes
    :param test_episodes: a list of full file names of the episodes we want to use as test data
    :param test_dir: a path to the dir where we will save the test data
    :param train_dir: a path to the dir where we will save the train data
    :return: iterate over all files in the subject's directory, and divide the episodes to train and test dirs
    according to the test_episodes list (should contain the names of the episodes that we want to use as test data).
    """
    for filename in os.listdir(subject_dir):
        if os.path.isfile(os.path.join(subject_dir, filename)) and filename in test_episodes:
            shutil.copy(os.path.join(subject_dir, filename), os.path.join(test_dir, filename))
        elif os.path.isfile(os.path.join(subject_dir, filename)) and 'full' not in filename:    # skip the file contains all episodes ('sub005_full')
            shutil.copy(os.path.join(subject_dir, filename), os.path.join(train_dir, filename))

def merge_data(dir):
    """
    :param dir: full path
    :return: merge all csv files in the directory to one csv file and save it as full.csv in the same directory
    """
    filepaths = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')]
    df = pd.concat(map(pd.read_csv, filepaths))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.to_csv(os.path.join(dir, 'full.csv'), index=False)


if __name__ == "__main__":
    divide_episodes(subject_dir, test_episodes, test_dir, train_dir)
    merge_data(test_dir)
    merge_data(train_dir)














