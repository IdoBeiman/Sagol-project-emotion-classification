import os
import pandas as pd
from help_methods import *
from model_class import MLmodel

def extract_details_from_file_name(filename):
    layer = filename.split("sub")[0].split("activations")[1].replace("_","")
    operation = filename.split("operation")[1].split("origin")[0].replace("_","")
    return operation+layer

def run():
    sents = [predicted_sentiment]
    log,dest =init_analysis()
    try:
        rmses = pd.DataFrame(columns=MLmodel.get_models_names())
        for test_pod in get_podcasts_from_folder():
            for s in sents:
                tmp_Results_dir = f"{dest}/{s}/{extract_details_from_file_name(test_pod)}"
                os.makedirs(tmp_Results_dir)
        
                print_and_log(log,f"*************** {s} *******************")
        
                MLmodel.models =[]
                SNN   = MLmodel(n1=128,n2=64,d_o=0.6,ac_func="tanh",model_type='dense', name='SNNtanh')
                SNNrelu   = MLmodel(n1=128,n2=64,d_o=0.6,ac_func="relu",model_type='dense', name='SNNRelu')
                uniLSTM = MLmodel(n1=32,n2=20,d_o=0.3,ac_func="sigmoid",model_type='uniLSTM',name='uLSTM')
                BiLSTM  = MLmodel(n1=16,n2=16,d_o=0.4,ac_func="sigmoid",model_type='BiLSTM',name='BiLSTM')
                Linear = MLmodel(name="Linear")
                Baseline = MLmodel(name='BL')
                nn_models = [SNN,uniLSTM, BiLSTM,SNNrelu]
        
                print_and_log(log,f"{test_pod}")

                train_df, test_df = get_train_test_df(test_pod,sentiment=s)

                predictions = pd.DataFrame()
                predictions['Real']= test_df[s]

                Linear.fit_elastic(train_df)
                Linear.predict_elastic(test_df)

                Baseline.fit_baseline(train_df)
                Baseline.predict_baseline(test_df)


                for m in nn_models:
                    print(m.name)
                    m.fit_NN(train_df)
                    m.predict_NN(test_df)

                row = {'Story':test_pod}
                for model in MLmodel.models:
                    print_and_log(log, f"{model.__dict__}")
                    row[model.name] = model.calculate_error(test_df[s])
                    predictions[model.name] = model.predictions

                rmses = rmses.append(row, ignore_index=True)
                predictionsFileName = f"{tmp_Results_dir}/{trim_file_extension(test_pod)}_model_predictions.csv"
                predictions.to_csv(predictionsFileName, mode='w', header=True)
        
                rmses.to_csv(f"{tmp_Results_dir}/direct_model_comparison.csv", mode='w', header=True)
                # plot_model_comparison(tmp_Results_dir)
                # plot_predictions(predictionsFileName,tmp_Results_dir)

    except Exception as e:
        print_and_log(log,f'An error occured: {e}')
    finally:
        log.close()
    
    return dest

run()

