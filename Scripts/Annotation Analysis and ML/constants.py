
# "avg" / "bin" / "avg_diff"
import os


LABALING_METHOD = "mean"

# names of all the features in the full data dataframe
feat_vec = [str(i) for i in range(0, 768)]

# which sentiment are we learning on
predicted_sentiment = "Nostalgia"

all_podcasts = ['train_activations_layer_8_sub005_operation_last_word_origin_full']# activations input file names
podasts_for_train = all_podcasts


#root dir (C:\Users\Tamara\Downloads\seminar_scripts\scripts)
root = os.path.join(os.getcwd(),"\debug") 

# results dir
Results_dir = f"{root}/Results"

#embeddind_dir = f"{root}/embeddings_last_layer" 


# path of full data files (output of 'create_ML_input)
data_path = "./Tokenized_data"