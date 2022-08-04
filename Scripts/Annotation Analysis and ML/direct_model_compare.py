
import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from sklearn import linear_model
from help_methods import *
from math import sqrt
from sklearn.metrics import mean_squared_error



class aModel:

    models = []

    def __init__(self, n1=0,n2=0,d_o=0.1,ac_func="tanh",model_type="",n_epochs=8,name=""):
        self.n1 = n1
        self.n2 = n2
        self.d_o = d_o # dropout
        self.ac_func = ac_func
        self.model_type = model_type # can be: 'dense', 'uniLSTM' or 'biLSTM'
        self.n_epochs = n_epochs
        self.name = name
        aModel.models.append(self)

    def get_models_names():
        return [m.name for m in aModel.models]


    def init_model(self,X):
        new_model = Sequential()
        if (self.model_type=="dense"):
            new_model.add(Dense(self.n1, activation=self.ac_func))
            new_model.add(Dropout(self.d_o))
            new_model.add(Dense(self.n2, activation=self.ac_func))
        if (self.model_type=="uniLSTM"):
            new_model.add(LSTM(self.n1, stateful=True, return_sequences=True, activation=self.ac_func, batch_input_shape=(1, X.shape[1], X.shape[2])))
            new_model.add(Dropout(self.d_o))
            new_model.add(LSTM(self.n2, stateful=True, return_sequences=True, activation=self.ac_func))
        if (self.model_type=="BiLSTM"):
            new_model.add(Bidirectional(LSTM(self.n1, stateful=True, return_sequences=True, activation=self.ac_func),batch_input_shape=(1, X.shape[1], X.shape[2])))
            new_model.add(Dropout(self.d_o))
            new_model.add(Bidirectional(LSTM(self.n2, stateful=True, return_sequences=True, activation=self.ac_func)))
        new_model.add(Dense(1))
        new_model.compile(loss='mean_squared_error', optimizer='adam')
        return new_model

    def fit_NN(self, train_df,show_progress=False, n_epochs=8):
        y = train_df[LABALING_METHOD]
        X = train_df.drop([LABALING_METHOD], axis=1)
        X = X.values.reshape(X.shape[0], 1, X.shape[1])
        model = self.init_model(X)
        history = []
        for i in range(self.n_epochs):
            tmp_history = model.fit(X, np.asarray(y), epochs=1, batch_size=1, verbose=show_progress, shuffle=False)
            history.append(tmp_history.history)
            model.reset_states()
        self.model = model
        self.param_num = model.count_params()
        self.history = history

    def predict_NN(self, test_df):
        predictions = list()
        for i in range(len(test_df)):
            y = test_df[LABALING_METHOD]
            X =  test_df.drop([LABALING_METHOD], axis=1).loc[i]
            yhat = forecast_lstm(self.model, 1, X)
            predictions.append(yhat.item())

        self.predictions=predictions

    def calculate_error(self,y):
        return sqrt(mean_squared_error(y, self.predictions))

    def fit_baseline(self,train_df):
        mode = train_df[LABALING_METHOD].mode().loc[0]
        self.model = mode

    def predict_baseline(self, test_df):
        self.predictions = [self.model for i in range(len(test_df))]

    def fit_bayesRidge(self, train_df):
        model = linear_model.BayesianRidge(normalize = True)
        y = train_df[LABALING_METHOD]
        X = train_df.drop([LABALING_METHOD], axis=1)

        self.model = model.fit(X, y)

    def predict_bayesRidge(self, test_df):
        BRPrediction = self.model.predict(test_df.drop([LABALING_METHOD], axis=1))
        self.predictions = BRPrediction


def run():
    log,dest =init_analysis()
    #sents = ["sadness", "control","excitment","unpleasantness" ]
    sents = [prdicted_sentiment]
    try:
        for s in sents:
            tmp_Results_dir = f"{dest}/{s}"
            os.makedirs(tmp_Results_dir)
    
            print_and_log(log,f"*************** {s} *******************")
    
            n_epochs = 8
    
            SNN   = aModel(n1=128,n2=64,d_o=0.6,ac_func="tanh",model_type='dense', name='SNN')
            uniLSTM = aModel(n1=32,n2=20,d_o=0.3,ac_func="sigmoid",model_type='uniLSTM',name='uLSTM')
            BiLSTM  = aModel(n1=16,n2=16,d_o=0.4,ac_func="sigmoid",model_type='BiLSTM',name='BiLSTM')
            Linear = aModel(name="Linear")
            Baseline = aModel(name='BL')
            nn_models = [SNN,uniLSTM, BiLSTM]
            rmses = pd.DataFrame(columns=aModel.get_models_names())
    
            for test_pod in podasts_for_train:
                print_and_log(log,f"{test_pod}")
    
                train_df, test_df = get_train_test_df(test_pod,sentiment=s)
                
                predictions = pd.DataFrame()
                predictions['Real']= test_df[LABALING_METHOD]
    
                Linear.fit_bayesRidge(train_df)
                Linear.predict_bayesRidge(test_df)
    
                Baseline.fit_baseline(train_df)
                Baseline.predict_baseline(test_df)
    
    
                for m in nn_models:
                    print(m.name)
                    m.fit_NN(train_df)
                    m.predict_NN(test_df)
    
                row = {'Story':test_pod}
                for model in aModel.models:
                    print_and_log(log, f"{model.__dict__}")
                    row[model.name] = model.calculate_error(test_df[LABALING_METHOD])
                    predictions[model.name] = model.predictions
    
                rmses = rmses.append(row, ignore_index=True)
    
                predictions.to_csv(f"{tmp_Results_dir}/{test_pod}_model_predictions.csv", mode='w', header=True)
    
            rmses.to_csv(f"{tmp_Results_dir}/direct_model_comparison.csv", mode='w', header=True)
            plot_model_comparison(tmp_Results_dir)
            plot_predictions(tmp_Results_dir)

    except Exception as e:
        print_and_log(log,f'An error occured: {e}')
    finally:
        log.close()
    
    return dest

run()

