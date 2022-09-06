from ast import Import
from tkinter import Y
import tensorflow as tf
from help_methods import *
from model_class import *
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

def get_optimal_model_params(dataset,sent,model_type,opimizer_grid_search=False):
    # define the grid search parameters
    param_grid = get_grid__optimizer_params(model_type) if opimizer_grid_search else get_grid_params(model_type)
    y_values = dataset[sent]
    original_dataset=dataset.copy(deep=True)
    dataset.drop(sent, axis=1, inplace=True)
    grid_search_df_process(dataset,sent)
    X = dataset.values.reshape(dataset.shape[0], 1, dataset.shape[1])
    if model_type == 'dense':
        compiled_model =KerasRegressor(model= create_model_for_grid_dense,optimizer='adam',initializer='normal', dropout_rate = 0.6,activation="tanh",weight_constraint=1.0, input_shape = X.shape[1:], layer_1_neurons=128, layer_2_neurons=64,optimizer_grid_search=opimizer_grid_search)
        grid = GridSearchCV(estimator=compiled_model,n_jobs=-1,cv=split_data_using_cross_validation(original_dataset,sent,True), param_grid=param_grid,scoring="neg_root_mean_squared_error",error_score='raise')
    elif model_type == 'uniLSTM':
        compiled_model =KerasRegressor(model= create_model_for_grid_uniLSTM,dropout_rate = 0.6, input_shape_dim1 = X.shape[1],input_shape_dim2= X.shape[2])
        grid = GridSearchCV(estimator=compiled_model,n_jobs=-1,cv=split_data_using_cross_validation(original_dataset,sent),scoring="r2", param_grid=param_grid,error_score='raise')
    grid_result = grid.fit(X, np.asarray(y_values))
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_params_
