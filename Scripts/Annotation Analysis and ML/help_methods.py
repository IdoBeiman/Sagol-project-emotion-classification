import os
import numpy as np
import pandas as pd
from os import listdir
import matplotlib.pyplot  as plt
import seaborn as sns
from datetime import datetime
from constants import *

def filter_rows_without_sent_and_split_train_test(df):
    df = df.dropna(subset=[predicted_sentiment])
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.7
    train = df[msk]
    test = df[~msk]
    return train, test

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

def get_feat_and_label_per_pod(test_pod):
    full_df = pd.read_csv(os.path.join(data_path,f"{test_pod}"))
    pod_train, pod_test=filter_rows_without_sent_and_split_train_test(full_df)
    pod_col_names = [predicted_sentiment] + feat_vec
    return pod_train[pod_col_names], pod_test[pod_col_names]

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

def get_train_test_df(test_pod,sentiment=predicted_sentiment):
    train_df, test_df = get_feat_and_label_per_pod(test_pod)
    test_df = test_df.reset_index()
    train_df = train_df.reset_index()
    test_df.drop(["index"], axis=1, inplace=True)
    train_df.drop(["index"], axis=1, inplace=True)
    return train_df, test_df

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
def trim_file_extension(filename):
    return filename.split(".")[0]
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
