import pandas as pd

subjects = ['sub-005', 'sub-006', 'sub-007']
language_models = ['bert', 'gpt', 't5']
layers = [i for i in range(12)]

path_to_origin_folder = 'C:/Users/Yuli/Documents/Uni/tokenized-data/by-model'

file_format = '{path}/{model}/layer{l}/{op}/data_activations_layer_{l}_{sub}_operation_{op}_origin_full_with_context_model_{model}.csv'
merged_file_format = '{path}/data_activations_layer_{l}_{sub}_operation_{op}_origin_full_with_context_model_{model}.csv'

path_to_dest_folder = 'C:/Users/Yuli/Documents/Uni/tokenized-data/merged-all-sub'

operation = 'mean'


for model in language_models:
    for layer in layers:
        if model == 't5' and layer > 6:
            continue
        full_df = pd.DataFrame()
        for sub in subjects:
            sub_df = pd.read_csv(file_format.format(path=path_to_origin_folder, model=model, l=layer, op=operation, sub=sub))
            sub_df['episodeName'].update([f'{sub}-{val}' for val in sub_df['episodeName'].values])
            full_df = pd.concat([full_df, sub_df])
        full_df.to_csv(merged_file_format.format(path=path_to_dest_folder, model=model, l=layer, op=operation, sub='all_sub'), index=False)
print('done')


# for layer in layers:
#     full_df = pd.DataFrame()
#     for sub in subjects:
#         sub_df = pd.read_csv(
#             '{path}/data_activations_layer_{l}_{sub}_operation_{op}_origin_full_with_context_model_{model}.csv'.format(
#                 path='C:/Users/Yuli/Documents/Uni/tokenized-data/gpt-pre-train', model='gpt2-pretrained', l=layer, op=operation, sub=sub))
#         sub_df['episodeName'].update([f'{sub}-{val}' for val in sub_df['episodeName'].values])
#         full_df = pd.concat([full_df, sub_df])
#     full_df.to_csv(
#         merged_file_format.format(path=path_to_dest_folder, model='gpt2-pretrained', l=layer, op=operation, sub='all_sub'),
#         index=False)

