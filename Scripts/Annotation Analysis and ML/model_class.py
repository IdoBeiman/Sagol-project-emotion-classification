from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from help_methods import *
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
import numpy as np
from sklearn import linear_model, metrics
from math import sqrt
from keras import backend
from sklearn.metrics import mean_squared_error, r2_score

class MLmodel:

    models = []
    def __init__(self, n1=0,n2=0,d_o=0.1,ac_func="tanh",model_type="",n_epochs=8,name="",initializer="",weight_constraint=""):
        self.n1 = n1
        self.n2 = n2
        self.d_o = d_o # dropout
        self.ac_func = ac_func
        self.initializer=initializer
        self.model_type = model_type # can be: 'dense', 'uniLSTM' or 'biLSTM'
        self.n_epochs = n_epochs
        self.name = name
        self.weight_constraint=weight_constraint
        self.grid_params = get_grid_params(model_type)
        MLmodel.models.append(self)

    def get_models_names():
        return [m.name for m in MLmodel.models]


    def init_model(self,X):
        new_model = Sequential()
        if (self.model_type=="dense"):
            new_model.add(Dense(self.n1,kernel_initializer=self.initializer, activation=self.ac_func,kernel_constraint=MaxNorm(self.weight_constraint)))
            new_model.add(Dropout(self.d_o))
            new_model.add(Dense(self.n2,kernel_initializer=self.initializer, activation=self.ac_func))
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
        y = train_df[predicted_sentiment]
        X = train_df.drop([predicted_sentiment], axis=1)
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
            y = test_df[predicted_sentiment]
            X =  test_df.drop([predicted_sentiment], axis=1).loc[i]
            yhat = forecast_lstm(self.model, 1, X)
            if yhat is None:
                print ("none")
            predictions.append(yhat.item())

        self.predictions=predictions

    def calculate_error(self,y):
        return sqrt(mean_squared_error(y, self.predictions))
    def calculate_r_squared_error (self,y ):
        return r2_score(y, self.predictions)
    def fit_baseline(self,train_df):
        mode = train_df[predicted_sentiment].mode().loc[0]
        self.model = mode

    def predict_baseline(self, test_df):
        self.predictions = [self.model for i in range(len(test_df))]

    def fit_bayesRidge(self, train_df):
        model = linear_model.BayesianRidge(normalize = True)
        y = train_df[predicted_sentiment]
        X = train_df.drop([predicted_sentiment], axis=1)

        self.model = model.fit(X, y)
        
    def fit_elastic(self, train_df):
        model = linear_model.ElasticNet(normalize = True)
        y = train_df[predicted_sentiment]
        X = train_df.drop([predicted_sentiment], axis=1)

        self.model = model.fit(X, y)

    def predict_bayesRidge(self, test_df):
        BRPrediction = self.model.predict(test_df.drop([predicted_sentiment], axis=1))
        self.predictions = BRPrediction
    def predict_elastic(self, test_df):
        elasticPrediction = self.model.predict(test_df.drop([predicted_sentiment], axis=1))
        self.predictions = elasticPrediction
    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
def create_model_for_grid_dense(dropout_rate,activation,weight_constraint,optimizer,initializer,input_shape,optimizer_grid_search=False):
	# create model
    model = Sequential()
    model.add(Dense(128,kernel_initializer=initializer,input_shape=input_shape, activation=activation,  kernel_constraint=MaxNorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation=activation,kernel_initializer=initializer))
    model.add(Dense(1,kernel_initializer=initializer))
    if optimizer_grid_search == True:
        return model
    else:
        model.compile(loss="mean_squared_error", optimizer=optimizer,metrics=[rmse])
        return model
def create_model_for_grid_uniLSTM(dropout_rate,activation,input_shape_dim1,input_shape_dim2):
	# create model
    model = Sequential()
    model.add(LSTM(32, stateful=True, return_sequences=True, activation=activation,batch_input_shape=(1, input_shape_dim1,input_shape_dim2)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(20, stateful=True, return_sequences=True, activation=activation))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam",metrics=[rmse])
    return model