import os
import numpy as np
import pandas as pd
from os import listdir
import matplotlib.pyplot  as plt
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from datetime import datetime
from constants import *

def process_tokens_dataframe(file_path, sents, smoothed=False, filter_ones=True):
    df = pd.read_csv(os.path.join(data_path,f"{file_path}"),index_col=0)
    label_cols = df.columns.intersection(all_emotions)
    labels = df[label_cols]
    if smoothed:
        smooth_labels(labels, factor=0.1)
    df[label_cols] = labels
    filtered_df = df[df[sents].notnull().all(1)] # right now it checks that all the sentiments exist - will be changed to check that any of them exists
    if filter_ones == True:
        filtered_df = filtered_df[filtered_df[predicted_sentiment] != 1]
        # labels_to_bins(test_df)
    return filtered_df

def balance_data(df, sents, method=None):
    if method:
        X = df.drop(sents, axis=1)
        y = pd.DataFrame(df[sents])
        # print(f'initial value count: {y.value_counts()}')  # make log global in order to use print and log
        if method == 'over':
            ros = RandomOverSampler(random_state=0)
            X_res, y_res = ros.fit_resample(X, y)
        elif method == 'under':
            rus = RandomUnderSampler(random_state=0)
            X_res, y_res = rus.fit_resample(X, y)
        res = pd.concat([y_res, X_res], axis=1)
        # print(f'value count after {method} sampling: {y_res.value_counts()}')
        return res
    return df

def split_data_using_cross_validation(df, sentitment,n_splits=5, random_split=False):
    if n_splits == 1:
        groups = df["episodeName"].to_numpy()
        logo = LeaveOneGroupOut()
        return logo.split(df, df[sentitment],groups=groups)
    elif random_split == False:
        groups = df["episodeName"].to_numpy()
        logo = LeaveOneGroupOut()
        return logo.split(df, df[sentitment],groups=groups)
    else:
        Kfold = KFold(n_splits)
        return Kfold.split(df, df[sentitment])

def get_num_splits(df, random_split=False, k=5):
    if random_split == False:
        groups = df["episodeName"].to_numpy()
        logo = LeaveOneGroupOut()
        iterations = logo.get_n_splits(groups=groups)
        return iterations
    else:
        return k
def get_grid_params(model_type):
    if model_type == "":
        return None
    elif model_type == "dense" or model_type == "BiLSTM":
        # return {'model__layer_2_neurons':[16,32,64,128],'model__layer_1_neurons':[128,256.512], 'model__optimizer':['adam', 'sgd'], 'model__initializer': ['normal', 'uniform'],'model__activation' : ["sigmoid","tanh","relu"],'model__dropout_rate':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],'model__weight_constraint' : [1.0,2.0,3.0,4.0]}
        return {'model__optimizer':['adam'],'model__activation' : ["sigmoid","tanh","relu"],'model__dropout_rate':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    elif model_type == "uniLSTM":
        return {'model__optimizer':['adam'],'model__activation' : ["sigmoid","tanh","relu"],'model__dropout_rate':[0.2,0.3,0.4,0.5,0.6,0.7]}
def get_grid__optimizer_params(model_type):
    if model_type == "uniLSTM":
        return None
    elif model_type == "uniLSTM" or model_type == "dense" or model_type == "BiLSTM":
        return {'model__optimizer':['adam'],'optimizer__learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3]}

def init_analysis():
    now = datetime.now()
    now_formatted = now.strftime("%d-%m-%Y_%H-%M-%S")
    project_root = os.path.dirname(os.path.dirname(__file__))
    dest_dir = f"{project_root}/destination_folder/{now_formatted}"
    os.makedirs(dest_dir)

    ## init log file
    log = open(f"{dest_dir}/running_log.txt", "w+")
    log.write(now.strftime("%d-%m-%Y_%H-%M-%S")+"\n\n")
    log.write(f"Predicted sentiment: {predicted_sentiment}\n")
    log.write(f"Train list:\n {podasts_for_train}\n")
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

def get_podcasts_from_folder():
    all_files = listdir(data_path)    
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    return csv_files


def plot_model_comparison(predictions_dir):
    results = pd.read_csv(f"{predictions_dir}/direct_model_comparison.csv")
    results.drop(['Unnamed: 0'],axis=1,inplace=True)

    sns.set_theme(style="whitegrid", font="Times New Roman")
    ax = sns.boxplot(data=results.drop('Story',axis=1),palette="Set1")

    ax.axes.set_title("Model Comparison",fontsize=25)
    sns.despine(left=True, bottom=True)
    sns.set(rc={'figure.figsize':(8,6),"font.size":50})
    ax.set_xlabel("Models",size=18)
    ax.set_ylabel("Mean RMSE Per Story",size=18)
    ax.set(xlabel="Models", ylabel="Mean RMSE Per Podcast")
    sns.set(font_scale=2)

    fig = ax.get_figure()
    fig.savefig(f"{predictions_dir}/direct_model_comparison.png")

    print(results.mean(axis = 0))
    print(results.std(axis = 0))
    
    return
def calcualte_model_accuracy (real_values, predictions ):
    acc = accuracy_score(real_values,predictions)

def trim_file_extension(filename):
    return filename.split(".")[0]
    
def post_split_process(train_df, test_df,sentiment, filter_ones = True): # filtering out 1 ranking which means the sentiment didn't appear in the segment
    test_df.drop(["episodeName"], axis=1, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_df.drop(["episodeName"], axis=1, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    to_remove = all_emotions
    if sentiment in to_remove:
        to_remove.remove(sentiment)
    train_df.drop([col for col in train_df.columns if   col in to_remove], axis=1, inplace=True)
    test_df.drop([col for col in test_df.columns if col in to_remove ], axis=1, inplace=True)
    train_df = balance_data(train_df, sentiment, method=balance_method)

    return train_df,test_df
def grid_search_df_process(df,sentiment):
    df.drop(["episodeName"], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    to_remove = all_emotions
    if sentiment in to_remove:
        to_remove.remove(sentiment)
    df.drop([col for col in df.columns if   col in to_remove], axis=1, inplace=True)
def plot_predictions(prediction_file_name, result_dir):
    sns.set()
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.color_palette("bright")
    
    results = pd.read_csv(prediction_file_name)
    y = results['Real']
    results.drop(['Unnamed: 0','Real'],axis=1,inplace=True)
    models = [m for m in results.columns if m!="BL"]
    
    fig,a =  plt.subplots(2,2,figsize=(15, 5),gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    sns.despine(left=True, bottom=True)
    fig.suptitle(f'{prediction_file_name}')
    i = 0
    for ax in a.ravel():
        l1=ax.plot(y,label = "real labels")[0]
        l3=ax.plot(results['BL'],label='BL')[0]
        l2=ax.plot(results[models[i]],label=models[i])
        
        #ax.legend(fontsize=8)
        ax.set_ylim([0.5, 5.5]) 
        ax.set_title(f'{models[i]}')
        i+=1
    
    # Create the legend
    fig.legend([l1, l3,l2],     # The line objects
                labels=['Real','BL','Model for comparison'],   # The labels for each line
                loc="lower center", ncol=3)
    
    for axi in fig.get_axes():
        axi.label_outer()
        
    fig.savefig(f"{result_dir}/predictions.png")
def labels_to_bins(df): # one sentiment only
    bins = [0,3,6,8]
    labels = bins[1:]
    df[predicted_sentiment] = pd.cut(df[predicted_sentiment], bins=bins, labels=[1,2,3])

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels
