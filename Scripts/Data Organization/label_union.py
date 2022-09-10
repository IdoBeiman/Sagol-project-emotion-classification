import os
import pandas as pd


def label_union(sub_num):

    path = '/data/emotion_project/transcriptions/raw_labels/' + sub_num  # labeled segments (csv files)
    new_path = '/data/emotion_project/transcriptions/raw_labels_union/' + sub_num
    files = [file for file in os.listdir(path) if file.endswith(".csv")]
    dfs = dict()
    for file in files:
        segment = file.rsplit("_", maxsplit=1)[0]
        to_upper = segment.split('_')[1].upper()
        segment = sub_num + '_' + to_upper
        #segment = segment.rsplit("_", maxsplit=1)[0] + "_" + to_upper

        df = pd.read_csv(os.path.join(path, file), index_col=0)
        if segment not in dfs:
            dfs[segment] = df.drop(['actual_time'], axis=1)
        else:
            dfs[segment] = pd.merge(dfs[segment], df, on="target_time", how="outer").drop(['actual_time'], axis=1)

    for key in dfs:
        dfs[key].to_csv(f"{new_path}/{key}.csv")








