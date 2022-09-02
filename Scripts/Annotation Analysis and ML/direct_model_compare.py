import os
import pandas as pd
from help_methods import *
from model_class import MLmodel

def extract_details_from_file_name(filename):
    model= filename.split("model")[1].split(".")[0].replace("_","")
    sub ="sub"+ filename.split("operation")[0].split("sub")[1].replace("_","")
    layer = filename.split("sub")[0].split("activations")[1].replace("_","")
    operation = filename.split("operation")[1].split("origin")[0].replace("_","")
    return sub+"_"+model+"_"+operation+"_"+layer

def run():
    sents = [predicted_sentiment]
    log,dest =init_analysis()
    try:
        rmses = pd.DataFrame(columns=MLmodel.get_models_names())
        tmp_Results_dir = f"{dest}/{predicted_sentiment}" # one folder for all of the tests
        os.makedirs(tmp_Results_dir)
        for podcast in get_podcasts_from_folder():
            total_predictions_df=pd.DataFrame()
            podcast_df = process_tokens_dataframe(podcast,sents=sents)
            for s in sents:
                print_and_log(log,f"*************** {s} *******************")
    
                MLmodel.models =[]
                SNNtanh   = MLmodel(n1=128,n2=64,d_o=0.6,ac_func="tanh",model_type='dense', name='SNNtanh')
                SNNrelu   = MLmodel(n1=128,n2=64,d_o=0.6,ac_func="relu",model_type='dense', name='SNNrelu')
                uniLSTM = MLmodel(n1=32,n2=20,d_o=0.3,ac_func="sigmoid",model_type='uniLSTM',name='uLSTM')
                BiLSTM  = MLmodel(n1=16,n2=16,d_o=0.4,ac_func="sigmoid",model_type='BiLSTM',name='BiLSTM')
                Linear = MLmodel(name="Linear")
                Baseline = MLmodel(name='BL')
                nn_models = [SNNtanh,uniLSTM, BiLSTM,SNNrelu]
        
                print_and_log(log,f"{podcast}")
                iterations = get_num_splits(podcast_df)
                current_iteration=1
                row = {'Story':extract_details_from_file_name(podcast)}
                accumulatedData ={}
                for train_indexes, test_indexes in split_data_using_cross_validation(podcast_df, s):
                    train_split_df = podcast_df.iloc[train_indexes]
                    test_split_df = podcast_df.iloc[test_indexes]
                    train_split_df,test_split_df = post_split_process(train_split_df,test_split_df,s)
                    
                    total_predictions_df['Real'+ "_"+str(current_iteration)+" iteration"]= test_split_df[s]
                    Linear.fit_elastic(train_split_df)
                    Linear.predict_elastic(test_split_df)

                    Baseline.fit_baseline(train_split_df)
                    Baseline.predict_baseline(test_split_df)


                    for m in nn_models:
                        print(m.name)
                        m.fit_NN(train_split_df)
                        m.predict_NN(test_split_df)

                    for model in MLmodel.models:
                        print_and_log(log, f"{model.__dict__}")
                        if model.name in accumulatedData.keys():
                            accumulatedData[model.name] += model.calculate_error(test_split_df[s])
                        else:
                            accumulatedData[model.name]=0
                            accumulatedData[model.name] += model.calculate_error(test_split_df[s])
                        total_predictions_df[model.name+"_"+" iteration_"+str(current_iteration)] = pd.Series(model.predictions)
                    current_iteration +=1
                for model in MLmodel.models: # after we finished the cross validation iterations we will divide the accumlated error by the number of iterations
                    row[model.name] =  accumulatedData[model.name]/iterations
                rmses = rmses.append(row, ignore_index=True)
                predictionsFileName = f"{tmp_Results_dir}/{trim_file_extension(podcast)}_model_predictions.csv"
            total_predictions_df.to_csv(predictionsFileName, mode='w', header=True) # predictions per podcast
        
                # plot_model_comparison(tmp_Results_dir)
                # plot_predictions(predictionsFileName,tmp_Results_dir)
        rmses.to_csv(f"{tmp_Results_dir}/all_models_comparison.csv", mode='w', header=True) # merged csv for all of the podcasts
    except Exception as e:
        print_and_log(log,f'An error occured: {e}')
    finally:
        log.close()
    
    return dest

run()

