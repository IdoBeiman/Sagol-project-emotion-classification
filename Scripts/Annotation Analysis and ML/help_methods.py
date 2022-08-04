# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:05:39 2021

@author: Tamara
"""
import os
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
from datetime import datetime
from constants import *

def init_analysis():
    now = datetime.now()
    now_formatted = now.strftime("%d-%m-%Y_%H-%M-%S")
    project_root = os.path.dirname(os.path.dirname(__file__))
    dest_dir = f"{project_root}/destination_folder/{now_formatted}"
    os.makedirs(dest_dir)

    ## init log file
    log = open(f"{dest_dir}/running_log.txt", "w+")
    log.write(now.strftime("%d-%m-%Y_%H-%M-%S")+"\n\n")
    log.write(f"Labeling method: {LABALING_METHOD}\n")
    log.write(f"Predicted sentiment: {prdicted_sentiment}\n")
    log.write(f"Train list:\n {podasts_for_train}\n")
    return log, dest_dir


def print_and_log(log, text):
    print(text)
    log.write(text+"\n")

def get_feat_and_label_per_pod(pod,sentiment):
    full_df = pd.read_csv(os.path.join(data_path,f"{sentiment}_ML_input.csv"))
    pod_full = full_df[full_df["audio_name"] == pod]
    pod_col_names = [LABALING_METHOD] + feat_vec
    return pod_full[pod_col_names]

def forecast_lstm(model, batch_size, X):
    """ make a one-step forecast.
    - batch_size should be 1 (according to the article)
    - X is the test set without labels! """
    X = X.values.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


def get_train_test_df(test_pod,sentiment=prdicted_sentiment):
    train_pods = [p for p in podasts_for_train if p!=test_pod]
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for pod in podasts_for_train: 
        print(pod)
        if (pod==test_pod):
            test_df = get_feat_and_label_per_pod(pod,sentiment)
        else:
            df = get_feat_and_label_per_pod(pod,sentiment)
            train_df = pd.concat([train_df, df])
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

def plot_predictions(predictions_dir):
    sns.set()
    sns.set_theme(style="whitegrid", font="Times New Roman")
    sns.color_palette("bright")
    
    for pod in podasts_for_train:
    
        results = pd.read_csv(f"{predictions_dir}/{pod}_model_predictions.csv")
        y = results['Real']
        results.drop(['Unnamed: 0','Real'],axis=1,inplace=True)
        models = [m for m in results.columns if m!="BL"]
        
        fig,a =  plt.subplots(2,2,figsize=(15, 5),gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
        sns.despine(left=True, bottom=True)
        fig.suptitle(f'{pod}')
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
            
        fig.savefig(f"{predictions_dir}/{pod}_predictions.png")
