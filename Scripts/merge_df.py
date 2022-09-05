import pandas as pd

df005 = pd.read_csv('C:\\Users\\Yuli\\Documents\\Uni\\tokenized-data\\by-model\\gpt2\\layer11\\mean\\data_activations_layer_11_sub-005_operation_mean_origin_full_with_context_model_gpt.csv')
df006 = pd.read_csv('C:\\Users\\Yuli\\Documents\\Uni\\tokenized-data\\by-model\\gpt2\\layer11\\mean\\data_activations_layer_11_sub-006_operation_mean_origin_full_with_context_model_gpt.csv')
df007 = pd.read_csv('C:\\Users\\Yuli\\Documents\\Uni\\tokenized-data\\by-model\\gpt2\\layer11\\mean\\data_activations_layer_11_sub-007_operation_mean_origin_full_with_context_model_gpt.csv')

# df005['Subject'] = 'sub-005'
# df006['Subject'] = 'sub-006'
# df007['Subject'] = 'sub-007'
#
# df005['episodeName'].update(['SUB005-' + val for val in df005['episodeName'].values])
# df006['episodeName'].update(['SUB006-' + val for val in df006['episodeName'].values])
# df007['episodeName'].update(['SUB007-' + val for val in df007['episodeName'].values])

df = pd.concat([df005, df006, df007])
df.to_csv('C:\\Users\\Yuli\\Documents\\Uni\\tokenized-data\\by-model\\gpt2\\layer11\\mean\\data_activations_layer_11_all_sub_operation_mean_origin_full_with_context_model_gpt.csv', index=False)

