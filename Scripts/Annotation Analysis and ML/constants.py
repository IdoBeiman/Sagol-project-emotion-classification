
# "avg" / "bin" / "avg_diff"
LABALING_METHOD = "mean"



# names of all the features in the full data dataframe
feat_vec = [str(i) for i in range(0, 768)]

# which sentiment are we learning on
prdicted_sentiment = "Sadness"

#all_podcasts = ['Where_I_Came_from', 'Autumn_story', 'How_Do_I_Say_This', 'Revision_Quest', 'Super_Duper', 'Breakout_Star', 'Please_Re_Lease_Me', 'no_simple_math', 'Mom_Hey_Mom', 'Go_To_The_Mattresses', 'one_last_swirl', 'Heart_Of_Gold', 'Distnat_Replay', 'Next_Of_Kindle']
all_podcasts = ['result0001','story2']
podasts_for_train = all_podcasts


#root dir (C:\Users\Tamara\Downloads\seminar_scripts\scripts)
root = "D:/users/itamar/code_tamara/Debug"

# results dir
Results_dir = f"{root}/Results"

#embeddind_dir = f"{root}/embeddings_last_layer" 


# path of full data files (output of 'create_ML_input)
data_path = "./Tokenized_data"