import os
import pandas as pd

from help_methods import *
from model_class import MLmodel
from pre_test_model_evaluation import *
class DirectModelCompare:

    run_time_models = []

    def __init__(self):
        self.logger, self.result_dir = init_analysis()

    def run(self):

        try:

            all_files_model_eval = pd.DataFrame(columns=MLmodel.get_models_names())

            # get all csv files from the given folder in constants file
            for file in get_files_from_folder():

                print_and_log(self.logger, f"Processing {file}")

                current_file_model_eval = pd.DataFrame(columns=MLmodel.get_models_names())
                run_details = self.extract_details_from_file_name(file)

                current_file_result_dir = f'{self.result_dir}/{run_details}'
                os.makedirs(current_file_result_dir)

                for s in PREDICTED_SENTIMENTS:

                    current_sent_result_dir = f'{current_file_result_dir}/{s}'
                    os.makedirs(current_sent_result_dir)

                    current_sent_model_eval = pd.DataFrame(columns=MLmodel.get_models_names())
                    current_sent_predictions = pd.DataFrame()
                    curr_iter_preds = pd.DataFrame()
                    concatenated_preds = pd.DataFrame()
                    collect_target_values =  np.array([])

                    print_and_log(self.logger, f"*************** {s} ***************")

                    current_sent_df = process_tokens_df(file, sents=[s])

                    # init models
                    MLmodel.models = []
                    nn_models = []

                    for model in MODELS:

                        if model == "SNN":
                            SNN = MLmodel(n1=128,n2=64,d_o=0.6,ac_func="sigmoid",initializer='uniform',model_type='dense', name='SNN')
                            nn_models.append(SNN)
                            # grid_params=get_optimal_model_params(pre_processed_df,s,model,opimizer_grid_search=False) # not running grid search since those were found as best
                            # SNN = MLmodel(n1=128,n2=64,d_o=grid_params["model__dropout_rate"],ac_func=grid_params["model__activation"], weight_constraint=grid_params["model__weight_constraint"],model='dense', name='SNN')
                        elif model == "uniLSTM":
                            uniLSTM = MLmodel(n1=64,n2=20,d_o=0.5,n_epochs=100,ac_func="sigmoid",model_type='uniLSTM',name='uLSTM')
                            nn_models.append(uniLSTM)
                            # grid_params=get_optimal_model_params(pre_processed_df,s,model)
                            # uniLSTM = MLmodel(n1=32,n2=20,d_o=grid_params.dropout_rate,ac_func=grid_params.activation,model='uniLSTM',name='uLSTM')
                        elif model == "BiLSTM":
                            BiLSTM = MLmodel(n1=100,n2=100,n3=16,d_o=0.4,n_epochs=140,ac_func="sigmoid",model_type='BiLSTM',name='BiLSTM')
                            nn_models.append(BiLSTM)
                            # BiLSTM = MLmodel(n1=16,n2=16,d_o=grid_params.dropout_rate,ac_func=grid_params.activation,model='BiLSTM',name='BiLSTM')
                        elif model == "Linear":
                            Linear = MLmodel(name="Linear")
                        elif model == "Baseline":
                            Baseline = MLmodel(name='Baseline')
                        else:
                            print_and_log(f'Invalid model type: {model}')

                    row = {'story': f'{self.extract_details_from_file_name(file)}_{s}'}
                    accumulated_data = {}
                    num_iter = get_num_splits(current_sent_df)
                    current_iteration = 0
                    
                    for train_indexes, test_indexes in split_data_using_cross_validation(current_sent_df.copy(deep=True), s):
                        current_iteration += 1
                        print_and_log(self.logger, f"#{current_iteration} iteration")
                        train_split_df = current_sent_df.copy(deep=True).iloc[train_indexes]
                        test_split_df = current_sent_df.copy(deep=True).iloc[test_indexes]
                        train_split_df, test_split_df = post_split_process(train_split_df, test_split_df, s)
                        current_sent_predictions[f'real_{str(current_iteration)}_iteration'] = test_split_df[s]

                        if 'Linear' in MODELS:
                            Linear.fit_bayesRidge(train_split_df,s)
                            Linear.predict_bayesRidge(test_split_df)

                        if 'Baseline' in MODELS:
                            Baseline.fit_baseline(train_split_df, s)
                            Baseline.predict_baseline(test_split_df, s)

                        for model in nn_models:
                            model.fit_NN(train_split_df.copy(deep=True), s)
                            model.predict_NN(test_split_df.copy(deep=True), s)

                        # eval model performance
                        collect_target_values = np.append(collect_target_values,test_split_df[s].to_numpy())
                        for model in MLmodel.models:
                            test_split_df_copy = test_split_df.copy(deep=True)
                            if f'{model.name}_rmse' in accumulated_data.keys():
                                accumulated_data[f'{model.name}_rmse'] += model.calculate_error(test_split_df_copy[s])
                            else:
                                accumulated_data[f'{model.name}_rmse'] = 0
                                accumulated_data[f'{model.name}_rmse'] += model.calculate_error(test_split_df_copy[s])
                            if f'{model.name}_r_square' in accumulated_data.keys():
                                accumulated_data[f'{model.name}_r_square'] += model.calculate_r_square(test_split_df_copy[s])
                            else:
                                accumulated_data[f'{model.name}_r_square'] = 0
                                accumulated_data[f'{model.name}_r_square'] += model.calculate_r_square(test_split_df_copy[s])
                            curr_iter_preds[f'{model.name}'] = pd.Series(model.predictions)
                            current_sent_predictions[f'{model.name}_iteration_{str(current_iteration)}'] = pd.Series(model.predictions)
                        concatenated_preds = pd.concat([concatenated_preds,curr_iter_preds])
                    
                    for model in MLmodel.models:
                    # after we finished the cross validation iterations we will divide the accumlated error
                    # by the number of iterations
                        row[f'{model.name}_rmse'] = accumulated_data[f'{model.name}_rmse'] / num_iter
                        row[f'{model.name}_r_square'] = accumulated_data[f'{model.name}_r_square'] / num_iter
                        concatenated_preds['Target'] =pd.Series(collect_target_values)
                        concatenated_preds[f'{model.name}_rmse'] =calculate_rmse_error( concatenated_preds["Target"] , concatenated_preds[f'{model.name}'])
                        row[f'{model.name}pearson_correl'] =calculate_pearson_correl( concatenated_preds["Target"] , concatenated_preds[f'{model.name}'])
                        row[f'{model.name}r_2_correl'] =calculate_r2_correl( concatenated_preds["Target"] , concatenated_preds[f'{model.name}'])
                    current_sent_model_eval = current_sent_model_eval.append(row, ignore_index=True)
                    current_file_model_eval = current_file_model_eval.append(row, ignore_index=True)
                    all_files_model_eval = all_files_model_eval.append(row, ignore_index=True)
                    concatenated_preds_path = f"{current_sent_result_dir}/{run_details}_{s}_concatenated_model_predictions.csv"
                    concatenated_preds.to_csv(concatenated_preds_path)
                    predictions_file_name = f"{current_sent_result_dir}/{run_details}_{s}_model_predictions.csv"
                    current_sent_predictions.to_csv(predictions_file_name, mode='w', header=True)
                    current_sent_model_eval.to_csv(f"{current_sent_result_dir}/{run_details}_{s}_models_comparison.csv", mode='w', header=True)

                current_file_model_eval.to_csv(f'{current_file_result_dir}/{run_details}_all_sentiments_models_comparison.csv', mode='w', header=True)

            all_files_model_eval.to_csv(f"{self.result_dir}/all_files_models_comparison.csv", mode='w', header=True)

        except Exception as e:
            print_and_log(self.logger, f'An error occurred: {e}')

        finally:
            self.logger.close()

    def extract_details_from_file_name(self, filename):
        model = filename.split('model_')[1].split('.')[0]
        sub = f'sub-{filename.split("operation")[0].split("sub_")[1]}'
        if sub == 'sub-':
            sub = 'all-subs'
        layer = f'layer-{filename.split("layer_")[1].split("_")[0]}'
        operation = filename.split("operation_")[1].split("_")[0]
        return f'{model}_{layer}_{operation}_{sub}'


if __name__ == '__main__':
    DirectModelCompare().run()

