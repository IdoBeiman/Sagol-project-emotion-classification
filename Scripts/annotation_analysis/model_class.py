from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from help_methods import *
from keras import callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow import keras
import numpy as np
from sklearn import linear_model, metrics
from math import sqrt
from keras import backend, metrics
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


class MLmodel:

    models = []

    def __init__(self, n1=0,n2=0,n3=0,d_o=0.1,ac_func="tanh",model_type="",n_epochs=EPOCH,name="",initializer="",weight_constraint=""):
        self.n1 = n1
        self.n2 = n2
        self.n3=n3
        self.d_o = d_o # dropout
        self.ac_func = ac_func
        self.initializer=initializer
        self.model_type = model_type # can be: 'dense', 'uniLSTM' or 'biLSTM'
        self.n_epochs = n_epochs
        self.name = name
        self.weight_constraint=weight_constraint
        self.grid_params = get_grid_params(model_type)
        self.predictions = list()
        MLmodel.models.append(self)

    def get_models_names():
        return [m.name for m in MLmodel.models]

    def init_model(self,X):
        new_model = Sequential()
        if (self.model_type=="dense"):
            new_model.add(Dense(self.n1,kernel_initializer=self.initializer, activation=self.ac_func))
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
        opt = keras.optimizers.Adam(learning_rate=0.001)
        new_model.compile(loss=rmse, optimizer=opt, metrics=[metrics.RootMeanSquaredError()])
        return new_model

    def fit_NN(self, train_df, sent, show_progress=True):
        # earlystopping = callbacks.EarlyStopping(monitor="loss", mode="min", patience=10, restore_best_weights=True)
        X = train_df.drop([sent], axis=1)
        y = train_df[sent]
        X = X.values.reshape(X.shape[0], 1, X.shape[1])
        model = self.init_model(X)
        if(self.model_type == "BiLSTM" or self.model_type == "uniLSTM"):
            for i in range(self.n_epochs):
                print (f"epoch {i+1} of {self.n_epochs}")
                model.fit(X, np.asarray(y), epochs=1, batch_size=1, verbose=show_progress, shuffle=False)
                model.reset_states()
        else:
            model.fit(X, np.asarray(y), epochs=self.n_epochs, batch_size=1, verbose=show_progress, shuffle=False)
        self.model = model
        self.param_num = model.count_params()

    def predict_NN(self, test_df, sent, batch_size=1):
        predictions = list()
        for i in range(len(test_df)):
            X = test_df.drop([sent], axis=1).iloc[i]
            yhat = forecast_lstm(self.model, batch_size, X)
            if yhat is None:
                print("none")
            predictions.append(yhat.item())
        self.predictions = predictions

    def calculate_error(self, y):
        return sqrt(mean_squared_error(y, self.predictions))

    def calculate_r_square(self, y):
        return r2_score(y, self.predictions)
        
    def calculate_mae(self, y):
        return mean_absolute_error(y, self.predictions)

    def fit_baseline(self, train_df, sent):
        mode = train_df[sent].mode().loc[0]
        self.model = mode

    def predict_baseline(self, test_df, sent):
        self.predictions = [self.model for i in range(len(test_df))]

    def fit_bayesRidge(self, train_df, sent):
        df = train_df.copy(deep=True)
        model = linear_model.BayesianRidge(normalize=True)
        y = df[sent]
        X = df.drop([s for s in PREDICTED_SENTIMENTS if s in df.columns], axis=1)
        self.model = model.fit(X, y)
        
    def fit_elastic(self, train_df, sent):
        df = train_df.copy(deep=True)
        model = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
        y = df[sent]
        X = df.drop([s for s in PREDICTED_SENTIMENTS if s in df.columns], axis=1)
        self.model = model.fit(X, y)

    def predict_bayesRidge(self, test_df, sent):
        df = test_df.copy(deep=True)
        X = df.drop([s for s in PREDICTED_SENTIMENTS if s in df.columns], axis=1)
        BRPrediction = self.model.predict(X)
        self.predictions = BRPrediction

    def predict_elastic(self, test_df, sent):
        df = test_df.copy(deep=True)
        elasticPredictions=[]
        for index,row in df.iterrows(): 
            row = row[:len(row)-1]
            elasticPredictions.append(self.model.predict([row]))
        self.predictions = elasticPredictions

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def create_model_for_grid_dense(dropout_rate,activation,optimizer,initializer,input_shape, layer_1_neurons, layer_2_neurons,optimizer_grid_search=False):
	# create model
    model = Sequential()
    model.add(Dense(layer_1_neurons,kernel_initializer=initializer,input_shape=input_shape, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(layer_2_neurons, activation=activation,kernel_initializer=initializer))
    model.add(Dense(1,kernel_initializer=initializer))
    model.compile(loss=rmse, optimizer=optimizer,metrics=[rmse])
    return model

def create_model_for_grid_uniLSTM(dropout_rate,activation,input_shape_dim1,input_shape_dim2):
	# create model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation=activation,batch_input_shape=(1, input_shape_dim1,input_shape_dim2)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(20, return_sequences=True, activation=activation))
    model.add(Dense(1))
    model.compile(loss=rmse, optimizer="adam",metrics=[rmse])
    return model