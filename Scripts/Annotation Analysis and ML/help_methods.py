import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from constants import *

from os import listdir
from datetime import datetime
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import LeaveOneGroupOut, KFold, GroupKFold, StratifiedKFold


def process_tokens_df(file_path, sents):
    """
    :param file_path:
    :param sents:
    :return:
    """
    df = pd.read_csv(os.path.join(ML_INPUT_DIR, f"{file_path}"), index_col=0)
    label_cols = df.columns.intersection(ALL_SENTIMENTS)
    labels = df[label_cols]
    if SMOOTH_LABELS:
        smooth_labels(labels, factor=0.1)
    df[label_cols] = labels
    filtered_df = df[df[sents].notnull().all(1)] # right now it checks that all the sentiments exist - will be changed to check that any of them exists
    if FILTER_ONES:
        filtered_df = filtered_df[filtered_df[sents[0]] != 1]
        # filtered_df = filtered_df[filtered_df[sents] != 1]
        # labels_to_bins(test_df)
    return filtered_df


def balance_data(df, sents):
    if BALANCE_DATA:
        X = df.drop(sents, axis=1)
        y = pd.DataFrame(df[sents])
        # log - f'initial value count: {y.value_counts()}. performing {BALANCE_METHOD} sampling'
        if BALANCE_METHOD == 'over':
            # read more about sampling strategies here -
            # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
            ros = RandomOverSampler(random_state=0, sampling_strategy='auto')
            X_res, y_res = ros.fit_resample(X, y)
        elif BALANCE_METHOD == 'under':
            # read more about sampling strategies here -
            # https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
            rus = RandomUnderSampler(random_state=0, sampling_strategy='auto')
            X_res, y_res = rus.fit_resample(X, y)
        res = pd.concat([y_res, X_res], axis=1)
        # log - f'value count after {method} sampling: {y_res.value_counts()}'
        return res
    return df


def concat_cv_results(results):
    try:
        results.drop(['Unnamed: 0'], axis=1, inplace=True)
    except Exception:
        pass
    iter_num = int(len(results.columns) / 6)
    df = results.iloc[:, 0:6]
    df.columns = [col.split('_')[0] for col in df.columns]
    for i in range(1, iter_num):
        new_df = results.iloc[:, 0+6*i:6+6*i]
        new_df.columns = [col.split('_')[0] for col in new_df.columns]
        df = pd.concat([df, new_df], ignore_index=True)
    return df


# def single_split_iterator(df):
#     indices = random.sample(range(0, len(df['episodeName'].values)), len((df['episodeName'].values))/5)
#     test_idx = np.where(df["episodeName"] == episode_name)
#     train_idx = np.where(df["episodeName"] != episode_name)
#     yield (train_idx,test_idx)


def targets_to_bins(df,sentiment):
    bins = [0,3,5,7]
    labels = [1.0,2.0,3.0]
    df[sentiment] = pd.cut(df[sentiment], bins=bins, labels=labels)
    return df


def split_data_using_cross_validation(df, sentitment, n_splits=N_SPLITS, split_type=CV_SPLIT_METHOD):
    df_copy = df.copy(deep=True)
    if split_type == 'GroupKfold':
        # log
        groups = df_copy["episodeName"].to_numpy()
        logo = GroupKFold(n_splits)
        return logo.split(df_copy, df_copy[sentitment], groups=groups)
    elif split_type == 'StratifiedKFold':
        # log
        skf = StratifiedKFold(n_splits=n_splits)
        return skf.split(df_copy, df_copy.loc[:,sentitment])
    elif split_type == 'Random':
        # log
        Kfold = KFold(n_splits)
        return Kfold.split(df_copy, df_copy[sentitment])
    else:
        # log!
        print('choose valid split method')
        return


def get_num_splits(df):
    if CV_SPLIT_METHOD == 'GroupKfold':
        groups = df["episodeName"].to_numpy()
        logo = GroupKFold(N_SPLITS)
        iterations = logo.get_n_splits(groups=groups)
        return iterations
    return N_SPLITS


def init_analysis():
    now = datetime.now()
    now_formatted = now.strftime("%d-%m-%Y_%H-%M-%S")
    project_root = os.path.dirname(os.path.dirname(__file__))
    dest_dir = f"{project_root}/destination_folder/{now_formatted}"
    os.makedirs(dest_dir)

    ## init log file
    log = open(f"{dest_dir}/running_log.txt", "w+")
    log.write(now.strftime("%d-%m-%Y_%H-%M-%S")+"\n\n")
    # add start logs with configuration
    return log, dest_dir


def print_and_log(log, text):
    print(text)
    log.write(text+"\n")


def forecast_lstm(model, batch_size, X):
    """ make a one-step forecast.
    - batch_size should be 1 (according to the article)
    - X is the test set without labels! """
    X = X.values.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


def get_files_from_folder():
    all_files = listdir(ML_INPUT_DIR)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    return csv_files


def trim_file_extension(filename):
    return filename.split(".")[0]
    

def post_split_process(train_df, test_df, sentiment, filter_ones = True): # filtering out 1 ranking which means the sentiment didn't appear in the segment
    test_df.drop(["episodeName"], axis=1, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_df.drop(["episodeName"], axis=1, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    to_remove = [sent for sent in ALL_SENTIMENTS if sent != sentiment]
    # if sentiment in to_remove:
    #     to_remove.remove(sentiment)
    train_df.drop([col for col in train_df.columns if col in to_remove], axis=1, inplace=True)
    test_df.drop([col for col in test_df.columns if col in to_remove], axis=1, inplace=True)
    train_df = balance_data(train_df, sentiment)
    return train_df, test_df


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels

# NEED IDO
def labels_to_bins(df): # one sentiment only
    bins = [0,3,6,8]
    labels = bins[1:]
    df[PREDICTED_SENTIMENT] = pd.cut(df[PREDICTED_SENTIMENT], bins=bins, labels=[1, 2, 3])

# NEED IDO
def get_grid_params(model_type):
    if model_type == "":
        return None
    elif model_type == "dense" or model_type == "BiLSTM":
        # return {'model__layer_2_neurons':[16,32,64,128],'model__layer_1_neurons':[128,256.512], 'model__optimizer':['adam', 'sgd'], 'model__initializer': ['normal', 'uniform'],'model__activation' : ["sigmoid","tanh","relu"],'model__dropout_rate':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],'model__weight_constraint' : [1.0,2.0,3.0,4.0]}
        return {'model__optimizer':['adam'],'model__activation' : ["sigmoid","tanh","relu"],'model__dropout_rate':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    elif model_type == "uniLSTM":
        return {'model__optimizer':['adam'],'model__activation' : ["sigmoid","tanh","relu"],'model__dropout_rate':[0.2,0.3,0.4,0.5,0.6,0.7]}


# NEED IDO
def get_grid__optimizer_params(model_type):
    if model_type == "uniLSTM":
        return None
    elif model_type == "uniLSTM" or model_type == "dense" or model_type == "BiLSTM":
        return {'model__optimizer':['adam'],'optimizer__learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3]}

# NEED IDO
def grid_search_df_process(df, sentiment):
    df.drop(["episodeName"], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    to_remove = ALL_SENTIMENTS
    if sentiment in to_remove:
        to_remove.remove(sentiment)
    df.drop([col for col in df.columns if col in to_remove], axis=1, inplace=True)
